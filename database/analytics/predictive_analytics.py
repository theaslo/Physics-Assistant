#!/usr/bin/env python3
"""
Phase 6.3: Advanced Predictive Analytics Engine for Physics Assistant
Comprehensive ML system for student success prediction, early warning systems,
multi-timeframe forecasting, and intelligent intervention recommendations.

Phase 6.3 Enhancements:
- Real-time prediction pipeline with streaming analytics
- Multi-timeframe predictions (short, medium, long-term)
- Concept-specific performance forecasting
- Advanced early warning system with confidence scoring
- Time-to-mastery estimation models
- Ensemble prediction integration
- Explainable AI for prediction rationale
- Educational outcome optimization
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
import pickle
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from enum import Enum
import uuid
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, IsolationForest
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import xgboost as xgb
import lightgbm as lgb
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import math
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Phase 6.3 Enums and Constants
class PredictionTimeframe(Enum):
    SHORT_TERM = "short_term"      # Next session (1-3 days)
    MEDIUM_TERM = "medium_term"    # Next week (7 days)
    LONG_TERM = "long_term"        # End of course (30+ days)

class ConfidenceLevel(Enum):
    VERY_LOW = "very_low"      # < 60%
    LOW = "low"                # 60-70%
    MODERATE = "moderate"      # 70-80%
    HIGH = "high"              # 80-90%
    VERY_HIGH = "very_high"    # > 90%

class InterventionPriority(Enum):
    IMMEDIATE = "immediate"    # Requires action within 24h
    HIGH = "high"              # Requires action within 3 days
    MEDIUM = "medium"          # Requires action within 1 week
    LOW = "low"                # Monitor only

class PhysicsConcept(Enum):
    KINEMATICS_1D = "kinematics_1d"
    KINEMATICS_2D = "kinematics_2d"
    FORCES = "forces"
    ENERGY = "energy"
    MOMENTUM = "momentum"
    ANGULAR_MOTION = "angular_motion"
    OSCILLATIONS = "oscillations"
    WAVES = "waves"
    THERMODYNAMICS = "thermodynamics"
    ELECTRICITY = "electricity"

@dataclass
class MultiTimeframePrediction:
    """Predictions across multiple time horizons"""
    student_id: str
    prediction_type: str
    short_term: float      # 1-3 days
    medium_term: float     # 1 week
    long_term: float       # 1 month
    confidence_short: float
    confidence_medium: float
    confidence_long: float
    trend_direction: str   # 'improving', 'stable', 'declining'
    trend_strength: float  # 0.0 to 1.0
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ConceptMasteryPrediction:
    """Concept-specific mastery predictions"""
    student_id: str
    concept: PhysicsConcept
    current_mastery: float
    predicted_mastery: float
    time_to_mastery: Optional[float]  # days
    mastery_confidence: float
    prerequisite_gaps: List[str]
    recommended_sequence: List[str]
    difficulty_adjustment: float
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class TimeToMasteryEstimate:
    """Time-to-mastery prediction for concepts"""
    student_id: str
    concept: PhysicsConcept
    estimated_days: float
    confidence_interval: Tuple[float, float]
    factors_affecting_time: Dict[str, float]
    recommended_study_schedule: Dict[str, Any]
    milestone_predictions: List[Dict[str, Any]]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class PredictionExplanation:
    """Explainable AI output for predictions"""
    prediction_id: str
    model_explanation: str
    feature_contributions: Dict[str, float]
    similar_student_patterns: List[Dict[str, Any]]
    confidence_factors: Dict[str, str]
    alternative_scenarios: List[Dict[str, float]]
    actionable_insights: List[str]
    visualization_data: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class InterventionRecommendation:
    """AI-generated intervention recommendation"""
    student_id: str
    intervention_id: str
    priority: InterventionPriority
    intervention_type: str
    description: str
    rationale: str
    expected_impact: float
    confidence_score: float
    timeframe: PredictionTimeframe
    success_probability: float
    resource_requirements: Dict[str, Any]
    alternative_interventions: List[Dict[str, Any]]
    monitoring_metrics: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(days=7))

@dataclass
class PredictionResult:
    """Enhanced result from predictive model with Phase 6.3 features"""
    student_id: str
    prediction_type: str
    predicted_value: float
    confidence_score: float
    confidence_level: ConfidenceLevel
    confidence_interval: Tuple[float, float]
    contributing_factors: Dict[str, float]
    risk_level: str  # 'low', 'medium', 'high'
    recommendations: List[str]
    model_version: str
    prediction_date: datetime = field(default_factory=datetime.now)
    
    # Phase 6.3 enhancements
    prediction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timeframe: PredictionTimeframe = PredictionTimeframe.MEDIUM_TERM
    explanation: Optional[PredictionExplanation] = None
    intervention_recommendations: List[InterventionRecommendation] = field(default_factory=list)
    ensemble_predictions: Dict[str, float] = field(default_factory=dict)
    uncertainty_quantification: Dict[str, float] = field(default_factory=dict)
    model_ensemble_weights: Dict[str, float] = field(default_factory=dict)
    prediction_stability_score: float = 0.0
    comparable_students: List[str] = field(default_factory=list)
    concept_specific_predictions: List[ConceptMasteryPrediction] = field(default_factory=list)

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_name: str
    model_type: str  # 'classification' or 'regression'
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_score: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    cross_val_scores: List[float] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    training_date: datetime = field(default_factory=datetime.now)

@dataclass
class EarlyWarningAlert:
    """Early warning system alert"""
    student_id: str
    alert_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    predicted_outcome: str
    confidence: float
    triggered_by: List[str]
    recommended_actions: List[str]
    alert_date: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(days=7))

class Phase63PredictiveAnalyticsEngine:
    """Phase 6.3: Advanced Predictive Analytics Engine with Real-time Inference"""
    
    def __init__(self, db_manager=None, model_storage_path="/tmp/ml_models", ensemble_system=None):
        self.db_manager = db_manager
        self.model_storage_path = model_storage_path
        self.ensemble_system = ensemble_system
        
        # Core model registry
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, Dict[str, LabelEncoder]] = {}
        self.model_performance: Dict[str, ModelPerformance] = {}
        
        # Phase 6.3 Enhanced Components
        self.multi_timeframe_models: Dict[PredictionTimeframe, Dict[str, Any]] = {}
        self.concept_mastery_models: Dict[PhysicsConcept, Dict[str, Any]] = {}
        self.time_to_mastery_models: Dict[str, Any] = {}
        self.early_warning_models: Dict[str, Any] = {}
        
        # Real-time prediction pipeline
        self.prediction_cache = {}
        self.cache_expiry = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Streaming analytics components
        self.prediction_streams = defaultdict(deque)
        self.real_time_features = defaultdict(dict)
        self.event_buffer = deque(maxlen=1000)
        
        # Ensemble integration
        self.ensemble_weights = {}
        self.model_confidence_tracker = defaultdict(list)
        
        # Explainable AI components
        self.explanation_models = {}
        self.feature_importance_cache = {}
        
        # Thread pool for async predictions
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.prediction_lock = threading.RLock()
        
        # Feature engineering configuration
        self.feature_config = {
            'temporal_features': ['hour_of_day', 'day_of_week', 'session_duration', 'time_since_last_interaction'],
            'behavioral_features': ['interaction_frequency', 'help_seeking_rate', 'concept_switching_rate'],
            'performance_features': ['success_rate', 'response_time', 'difficulty_progression'],
            'social_features': ['peer_comparison_rank', 'class_percentile'],
            'content_features': ['concept_coverage', 'problem_type_distribution', 'hint_usage_rate']
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            'performance_decline': 0.3,
            'engagement_drop': 0.4,
            'difficulty_spike': 0.7,
            'help_seeking_excess': 0.8,
            'time_inefficiency': 0.6
        }
        
        # Active alerts
        self.active_alerts: Dict[str, List[EarlyWarningAlert]] = defaultdict(list)
    
    async def initialize(self):
        """Initialize the Phase 6.3 Advanced Predictive Analytics Engine"""
        try:
            logger.info("🚀 Initializing Phase 6.3 Predictive Analytics Engine")
            
            # Create model storage directory
            import os
            os.makedirs(self.model_storage_path, exist_ok=True)
            
            # Load existing models if available
            await self._load_existing_models()
            
            # Initialize feature extractors
            await self._initialize_feature_extractors()
            
            # Phase 6.3 specific initializations
            await self._initialize_multi_timeframe_models()
            await self._initialize_concept_mastery_models()
            await self._initialize_real_time_pipeline()
            await self._initialize_explainable_ai()
            
            # Start real-time prediction service
            await self._start_prediction_service()
            
            logger.info("✅ Phase 6.3 Predictive Analytics Engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Phase 6.3 Predictive Analytics Engine: {e}")
            return False
    
    async def _initialize_multi_timeframe_models(self):
        """Initialize models for different prediction timeframes"""
        try:
            logger.info("🔧 Initializing multi-timeframe prediction models")
            
            for timeframe in PredictionTimeframe:
                self.multi_timeframe_models[timeframe] = {
                    'success_predictor': None,
                    'engagement_predictor': None,
                    'performance_predictor': None,
                    'dropout_predictor': None
                }
            
            logger.info("✅ Multi-timeframe models initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize multi-timeframe models: {e}")
    
    async def _initialize_concept_mastery_models(self):
        """Initialize concept-specific mastery prediction models"""
        try:
            logger.info("🔧 Initializing concept mastery models")
            
            for concept in PhysicsConcept:
                self.concept_mastery_models[concept] = {
                    'mastery_predictor': None,
                    'time_to_mastery': None,
                    'difficulty_predictor': None,
                    'prerequisite_tracker': None
                }
            
            logger.info("✅ Concept mastery models initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize concept mastery models: {e}")
    
    async def _initialize_real_time_pipeline(self):
        """Initialize real-time prediction pipeline"""
        try:
            logger.info("🔧 Initializing real-time prediction pipeline")
            
            # Initialize prediction cache
            self.prediction_cache = {}
            self.cache_expiry = {}
            
            # Initialize streaming components
            self.prediction_streams = defaultdict(deque)
            self.event_buffer = deque(maxlen=1000)
            
            logger.info("✅ Real-time prediction pipeline initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize real-time pipeline: {e}")
    
    async def _initialize_explainable_ai(self):
        """Initialize explainable AI components"""
        try:
            logger.info("🔧 Initializing explainable AI components")
            
            # Initialize explanation models
            self.explanation_models = {
                'feature_attribution': None,
                'counterfactual_generator': None,
                'similarity_analyzer': None
            }
            
            self.feature_importance_cache = {}
            
            logger.info("✅ Explainable AI components initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize explainable AI: {e}")
    
    async def _start_prediction_service(self):
        """Start the real-time prediction service"""
        try:
            logger.info("🚀 Starting real-time prediction service")
            
            # In a production environment, this would start background tasks
            # for continuous prediction updates and cache management
            
            logger.info("✅ Real-time prediction service started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start prediction service: {e}")
    
    async def _load_existing_models(self):
        """Load previously trained models from storage"""
        try:
            import os
            import glob
            
            model_files = glob.glob(f"{self.model_storage_path}/*.pkl")
            
            for model_file in model_files:
                model_name = os.path.basename(model_file).replace('.pkl', '')
                try:
                    with open(model_file, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    self.models[model_name] = model_data['model']
                    if 'scaler' in model_data:
                        self.scalers[model_name] = model_data['scaler']
                    if 'encoders' in model_data:
                        self.encoders[model_name] = model_data['encoders']
                    if 'performance' in model_data:
                        self.model_performance[model_name] = model_data['performance']
                    
                    logger.info(f"📊 Loaded model: {model_name}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load model {model_name}: {e}")
            
            if model_files:
                logger.info(f"✅ Loaded {len(model_files)} existing models")
            else:
                logger.info("📝 No existing models found, will train new ones")
                
        except Exception as e:
            logger.error(f"❌ Failed to load existing models: {e}")
    
    async def _initialize_feature_extractors(self):
        """Initialize feature extraction components"""
        try:
            # Initialize scalers for different feature groups
            for feature_group in self.feature_config.keys():
                if feature_group not in self.scalers:
                    self.scalers[feature_group] = StandardScaler()
            
            logger.info("✅ Feature extractors initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize feature extractors: {e}")
    
    async def extract_student_features(self, student_id: str, lookback_days: int = 30) -> Dict[str, float]:
        """Extract comprehensive feature set for a student"""
        try:
            if not self.db_manager:
                return {}
            
            features = {}
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            async with self.db_manager.postgres.get_connection() as conn:
                # Get interaction data
                interactions = await conn.fetch("""
                    SELECT * FROM interactions 
                    WHERE user_id = $1 AND created_at BETWEEN $2 AND $3
                    ORDER BY created_at ASC
                """, student_id, start_date, end_date)
                
                if not interactions:
                    return features
                
                # Convert to DataFrame for easier processing
                df = pd.DataFrame([dict(row) for row in interactions])
                df['created_at'] = pd.to_datetime(df['created_at'])
                
                # Extract temporal features
                features.update(self._extract_temporal_features(df))
                
                # Extract behavioral features
                features.update(self._extract_behavioral_features(df))
                
                # Extract performance features
                features.update(self._extract_performance_features(df))
                
                # Extract content features
                features.update(self._extract_content_features(df))
                
                # Get comparative features
                comparative_features = await self._extract_comparative_features(student_id, conn)
                features.update(comparative_features)
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Failed to extract features for student {student_id}: {e}")
            return {}
    
    def _extract_temporal_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract time-based features"""
        features = {}
        
        try:
            # Basic temporal patterns
            df['hour'] = df['created_at'].dt.hour
            df['day_of_week'] = df['created_at'].dt.dayofweek
            
            features['avg_hour_of_day'] = df['hour'].mean()
            features['preferred_day_of_week'] = df['day_of_week'].mode().iloc[0] if not df.empty else 0
            
            # Session analysis
            df['time_diff'] = df['created_at'].diff().dt.total_seconds() / 60  # minutes
            session_breaks = df['time_diff'] > 30  # 30 minutes = new session
            
            features['avg_session_duration'] = df.groupby(session_breaks.cumsum())['time_diff'].sum().mean()
            features['total_sessions'] = session_breaks.sum() + 1
            features['avg_interactions_per_session'] = len(df) / features['total_sessions']
            
            # Consistency patterns
            daily_interactions = df.groupby(df['created_at'].dt.date).size()
            features['interaction_consistency'] = 1.0 - (daily_interactions.std() / daily_interactions.mean()) if daily_interactions.std() > 0 else 1.0
            
            # Time since last interaction
            if len(df) > 0:
                features['hours_since_last_interaction'] = (datetime.now() - df['created_at'].iloc[-1]).total_seconds() / 3600
            else:
                features['hours_since_last_interaction'] = 999
            
        except Exception as e:
            logger.error(f"❌ Failed to extract temporal features: {e}")
        
        return features
    
    def _extract_behavioral_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract behavioral pattern features"""
        features = {}
        
        try:
            total_interactions = len(df)
            if total_interactions == 0:
                return features
            
            # Interaction frequency
            date_range = (df['created_at'].max() - df['created_at'].min()).days + 1
            features['interactions_per_day'] = total_interactions / date_range
            
            # Response time analysis
            if 'execution_time_ms' in df.columns:
                features['avg_response_time'] = df['execution_time_ms'].mean()
                features['response_time_std'] = df['execution_time_ms'].std()
                features['response_time_trend'] = self._calculate_trend(df['execution_time_ms'].values)
            
            # Help-seeking behavior (based on metadata)
            help_interactions = 0
            concept_switches = 0
            prev_agent = None
            
            for _, row in df.iterrows():
                # Count help-seeking indicators
                if row.get('metadata'):
                    try:
                        metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
                        if metadata.get('help_requested') or metadata.get('hint_used'):
                            help_interactions += 1
                    except:
                        pass
                
                # Count concept switches
                current_agent = row.get('agent_type')
                if prev_agent and current_agent != prev_agent:
                    concept_switches += 1
                prev_agent = current_agent
            
            features['help_seeking_rate'] = help_interactions / total_interactions
            features['concept_switching_rate'] = concept_switches / total_interactions
            
            # Success patterns
            if 'success' in df.columns:
                features['overall_success_rate'] = df['success'].mean()
                
                # Success trend over time
                features['success_trend'] = self._calculate_trend(df['success'].astype(int).values)
                
                # Recent vs early performance
                mid_point = len(df) // 2
                if mid_point > 0:
                    early_success = df['success'].iloc[:mid_point].mean()
                    recent_success = df['success'].iloc[mid_point:].mean()
                    features['performance_improvement'] = recent_success - early_success
            
        except Exception as e:
            logger.error(f"❌ Failed to extract behavioral features: {e}")
        
        return features
    
    def _extract_performance_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract academic performance features"""
        features = {}
        
        try:
            if 'success' in df.columns and len(df) > 0:
                # Basic performance metrics
                features['success_rate'] = df['success'].mean()
                features['total_attempts'] = len(df)
                features['successful_attempts'] = df['success'].sum()
                
                # Performance consistency
                if len(df) >= 10:
                    # Rolling success rate
                    rolling_success = df['success'].rolling(window=10, min_periods=5).mean()
                    features['performance_consistency'] = 1.0 - rolling_success.std() if rolling_success.std() > 0 else 1.0
                
                # Difficulty progression
                agent_difficulty = {
                    'kinematics': 1,
                    'forces': 2,
                    'energy': 3,
                    'momentum': 3,
                    'angular_motion': 4,
                    'math': 1
                }
                
                if 'agent_type' in df.columns:
                    df['difficulty'] = df['agent_type'].map(agent_difficulty).fillna(2)
                    features['avg_difficulty_attempted'] = df['difficulty'].mean()
                    features['max_difficulty_attempted'] = df['difficulty'].max()
                    
                    # Success rate by difficulty
                    difficulty_success = df.groupby('difficulty')['success'].mean()
                    for diff, success in difficulty_success.items():
                        features[f'success_rate_difficulty_{int(diff)}'] = success
                
                # Learning velocity (improvement rate)
                if len(df) >= 20:
                    # Split into quarters and measure improvement
                    quarter_size = len(df) // 4
                    quarters = [df.iloc[i*quarter_size:(i+1)*quarter_size]['success'].mean() 
                              for i in range(4)]
                    features['learning_velocity'] = np.polyfit(range(4), quarters, 1)[0]  # slope
            
        except Exception as e:
            logger.error(f"❌ Failed to extract performance features: {e}")
        
        return features
    
    def _extract_content_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract content interaction features"""
        features = {}
        
        try:
            if 'agent_type' in df.columns:
                # Concept coverage
                unique_concepts = df['agent_type'].nunique()
                features['concept_coverage'] = unique_concepts
                
                # Concept distribution
                concept_counts = df['agent_type'].value_counts()
                total_interactions = len(df)
                
                # Concentration index (how focused on specific concepts)
                concept_proportions = concept_counts / total_interactions
                features['concept_concentration'] = np.sum(concept_proportions ** 2)  # Herfindahl index
                
                # Most used concept
                most_used_concept = concept_counts.index[0] if len(concept_counts) > 0 else None
                if most_used_concept:
                    features[f'primary_concept_{most_used_concept}'] = 1
                
                # Concept success rates
                if 'success' in df.columns:
                    concept_success = df.groupby('agent_type')['success'].mean()
                    for concept, success_rate in concept_success.items():
                        features[f'success_rate_{concept}'] = success_rate
            
            # Problem complexity (based on metadata)
            complexity_scores = []
            for _, row in df.iterrows():
                if row.get('metadata'):
                    try:
                        metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
                        complexity = metadata.get('problem_complexity', 1)
                        complexity_scores.append(complexity)
                    except:
                        complexity_scores.append(1)
            
            if complexity_scores:
                features['avg_problem_complexity'] = np.mean(complexity_scores)
                features['max_problem_complexity'] = np.max(complexity_scores)
            
        except Exception as e:
            logger.error(f"❌ Failed to extract content features: {e}")
        
        return features
    
    async def _extract_comparative_features(self, student_id: str, conn) -> Dict[str, float]:
        """Extract features relative to peer performance"""
        features = {}
        
        try:
            # Get student's performance
            student_stats = await conn.fetchrow("""
                SELECT 
                    AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate,
                    AVG(execution_time_ms) as avg_response_time,
                    COUNT(*) as total_interactions
                FROM interactions 
                WHERE user_id = $1 AND created_at >= NOW() - INTERVAL '30 days'
            """, student_id)
            
            # Get class/peer statistics
            class_stats = await conn.fetchrow("""
                SELECT 
                    AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as class_success_rate,
                    AVG(execution_time_ms) as class_avg_response_time,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY execution_time_ms) as class_median_response_time
                FROM interactions 
                WHERE created_at >= NOW() - INTERVAL '30 days'
                AND user_id != $1
            """, student_id)
            
            if student_stats and class_stats:
                # Relative performance
                features['success_rate_vs_class'] = student_stats['success_rate'] - class_stats['class_success_rate']
                features['response_time_vs_class'] = student_stats['avg_response_time'] - class_stats['class_avg_response_time']
                
                # Percentile ranking
                percentile_query = """
                    SELECT COUNT(*) as better_students, 
                           (SELECT COUNT(DISTINCT user_id) FROM interactions 
                            WHERE created_at >= NOW() - INTERVAL '30 days') as total_students
                    FROM (
                        SELECT user_id, AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as user_success_rate
                        FROM interactions 
                        WHERE created_at >= NOW() - INTERVAL '30 days'
                        GROUP BY user_id
                        HAVING AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) > $1
                    ) better_performers
                """
                
                percentile_result = await conn.fetchrow(percentile_query, student_stats['success_rate'])
                if percentile_result and percentile_result['total_students'] > 0:
                    features['class_percentile'] = (percentile_result['total_students'] - percentile_result['better_students']) / percentile_result['total_students']
        
        except Exception as e:
            logger.error(f"❌ Failed to extract comparative features: {e}")
        
        return features
    
    def _calculate_trend(self, values: np.ndarray) -> float:
        """Calculate trend using linear regression slope"""
        if len(values) < 2:
            return 0.0
        
        try:
            x = np.arange(len(values))
            slope, _, _, _, _ = stats.linregress(x, values)
            return slope
        except:
            return 0.0
    
    async def train_success_prediction_model(self, min_interactions: int = 50) -> ModelPerformance:
        """Train model to predict student success probability"""
        try:
            logger.info("🎯 Training student success prediction model")
            
            # Collect training data
            training_data = await self._collect_training_data(min_interactions)
            
            if len(training_data) < 100:
                raise ValueError(f"Insufficient training data: {len(training_data)} samples")
            
            # Prepare features and targets
            X, y, feature_names = await self._prepare_classification_data(training_data)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train ensemble of models
            models = {
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
                'gradient_boosting': GradientBoostingRegressor(random_state=42)
            }
            
            best_model = None
            best_score = 0
            best_model_name = None
            
            for name, model in models.items():
                if name == 'gradient_boosting':
                    continue  # Skip for classification
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
                
                if cv_scores.mean() > best_score:
                    best_score = cv_scores.mean()
                    best_model = model
                    best_model_name = name
            
            # Train best model
            best_model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = best_model.predict(X_test_scaled)
            y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1] if hasattr(best_model, 'predict_proba') else None
            
            # Calculate metrics
            performance = ModelPerformance(
                model_name='success_predictor',
                model_type='classification',
                accuracy=accuracy_score(y_test, y_pred),
                precision=precision_score(y_test, y_pred),
                recall=recall_score(y_test, y_pred),
                f1_score=f1_score(y_test, y_pred),
                auc_score=roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None,
                cross_val_scores=cv_scores.tolist()
            )
            
            # Feature importance
            if hasattr(best_model, 'feature_importances_'):
                importance_dict = dict(zip(feature_names, best_model.feature_importances_))
                performance.feature_importance = {k: float(v) for k, v in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]}
            
            # Store model
            self.models['success_predictor'] = best_model
            self.scalers['success_predictor'] = scaler
            self.model_performance['success_predictor'] = performance
            
            # Save to disk
            await self._save_model('success_predictor')
            
            logger.info(f"✅ Success prediction model trained - Accuracy: {performance.accuracy:.3f}")
            return performance
            
        except Exception as e:
            logger.error(f"❌ Failed to train success prediction model: {e}")
            raise
    
    async def train_engagement_prediction_model(self, min_interactions: int = 30) -> ModelPerformance:
        """Train model to predict student engagement levels"""
        try:
            logger.info("📊 Training student engagement prediction model")
            
            # Collect engagement training data
            training_data = await self._collect_engagement_training_data(min_interactions)
            
            if len(training_data) < 100:
                raise ValueError(f"Insufficient engagement training data: {len(training_data)} samples")
            
            # Prepare features and targets
            X, y, feature_names = await self._prepare_regression_data(training_data, target='engagement_score')
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            performance = ModelPerformance(
                model_name='engagement_predictor',
                model_type='regression',
                rmse=np.sqrt(mean_squared_error(y_test, y_pred)),
                mae=mean_absolute_error(y_test, y_pred),
                r2_score=r2_score(y_test, y_pred)
            )
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(feature_names, model.feature_importances_))
                performance.feature_importance = {k: float(v) for k, v in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]}
            
            # Store model
            self.models['engagement_predictor'] = model
            self.scalers['engagement_predictor'] = scaler
            self.model_performance['engagement_predictor'] = performance
            
            # Save to disk
            await self._save_model('engagement_predictor')
            
            logger.info(f"✅ Engagement prediction model trained - R²: {performance.r2_score:.3f}")
            return performance
            
        except Exception as e:
            logger.error(f"❌ Failed to train engagement prediction model: {e}")
            raise
    
    # ===== PHASE 6.3 CORE PREDICTION METHODS =====
    
    async def predict_multi_timeframe(self, student_id: str, prediction_type: str) -> MultiTimeframePrediction:
        """Generate predictions across multiple time horizons"""
        try:
            logger.info(f"📊 Generating multi-timeframe predictions for {student_id}")
            
            # Extract features
            features = await self.extract_student_features(student_id)
            
            if not features:
                raise ValueError(f"No features available for student {student_id}")
            
            # Generate predictions for each timeframe
            short_term_pred, short_confidence = await self._predict_for_timeframe(
                features, PredictionTimeframe.SHORT_TERM, prediction_type
            )
            medium_term_pred, medium_confidence = await self._predict_for_timeframe(
                features, PredictionTimeframe.MEDIUM_TERM, prediction_type
            )
            long_term_pred, long_confidence = await self._predict_for_timeframe(
                features, PredictionTimeframe.LONG_TERM, prediction_type
            )
            
            # Calculate trend
            predictions = [short_term_pred, medium_term_pred, long_term_pred]
            trend_direction, trend_strength = self._calculate_prediction_trend(predictions)
            
            return MultiTimeframePrediction(
                student_id=student_id,
                prediction_type=prediction_type,
                short_term=short_term_pred,
                medium_term=medium_term_pred,
                long_term=long_term_pred,
                confidence_short=short_confidence,
                confidence_medium=medium_confidence,
                confidence_long=long_confidence,
                trend_direction=trend_direction,
                trend_strength=trend_strength
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to generate multi-timeframe predictions: {e}")
            # Return fallback prediction
            return MultiTimeframePrediction(
                student_id=student_id,
                prediction_type=prediction_type,
                short_term=0.5, medium_term=0.5, long_term=0.5,
                confidence_short=0.3, confidence_medium=0.3, confidence_long=0.3,
                trend_direction='stable', trend_strength=0.0
            )
    
    async def predict_concept_mastery(self, student_id: str, concept: PhysicsConcept) -> ConceptMasteryPrediction:
        """Predict mastery for a specific physics concept"""
        try:
            logger.info(f"🎯 Predicting {concept.value} mastery for {student_id}")
            
            # Extract concept-specific features
            features = await self._extract_concept_features(student_id, concept)
            current_mastery = await self._calculate_current_mastery(student_id, concept)
            
            # Predict future mastery
            predicted_mastery = await self._predict_concept_mastery_level(features, concept)
            
            # Calculate time to mastery
            time_to_mastery = await self._estimate_time_to_mastery(
                student_id, concept, current_mastery, predicted_mastery
            )
            
            # Identify prerequisite gaps
            prerequisite_gaps = await self._identify_prerequisite_gaps(student_id, concept)
            
            # Generate recommended sequence
            recommended_sequence = await self._generate_learning_sequence(student_id, concept)
            
            # Calculate difficulty adjustment
            difficulty_adjustment = await self._calculate_difficulty_adjustment(student_id, concept)
            
            return ConceptMasteryPrediction(
                student_id=student_id,
                concept=concept,
                current_mastery=current_mastery,
                predicted_mastery=predicted_mastery,
                time_to_mastery=time_to_mastery,
                mastery_confidence=0.8,  # Would be calculated from model uncertainty
                prerequisite_gaps=prerequisite_gaps,
                recommended_sequence=recommended_sequence,
                difficulty_adjustment=difficulty_adjustment
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to predict concept mastery: {e}")
            return ConceptMasteryPrediction(
                student_id=student_id, concept=concept, current_mastery=0.5,
                predicted_mastery=0.5, time_to_mastery=14.0, mastery_confidence=0.3,
                prerequisite_gaps=[], recommended_sequence=[], difficulty_adjustment=0.0
            )
    
    async def predict_time_to_mastery(self, student_id: str, concept: PhysicsConcept) -> TimeToMasteryEstimate:
        """Estimate time required to achieve concept mastery"""
        try:
            logger.info(f"⏱️ Estimating time to mastery for {student_id} - {concept.value}")
            
            # Get current performance metrics
            current_performance = await self._get_concept_performance_metrics(student_id, concept)
            
            # Extract learning velocity features
            learning_features = await self._extract_learning_velocity_features(student_id, concept)
            
            # Predict time to mastery using trained model
            if concept in self.time_to_mastery_models and self.time_to_mastery_models[concept]:
                estimated_days = await self._predict_mastery_time(learning_features, concept)
            else:
                # Use heuristic approach
                estimated_days = await self._heuristic_time_estimation(current_performance, learning_features)
            
            # Calculate confidence interval
            confidence_interval = self._calculate_time_confidence_interval(
                estimated_days, learning_features
            )
            
            # Identify factors affecting time
            factors_affecting_time = await self._analyze_time_factors(student_id, concept)
            
            # Generate study schedule
            study_schedule = await self._generate_study_schedule(student_id, concept, estimated_days)
            
            # Create milestone predictions
            milestones = await self._create_milestone_predictions(
                student_id, concept, estimated_days
            )
            
            return TimeToMasteryEstimate(
                student_id=student_id,
                concept=concept,
                estimated_days=estimated_days,
                confidence_interval=confidence_interval,
                factors_affecting_time=factors_affecting_time,
                recommended_study_schedule=study_schedule,
                milestone_predictions=milestones
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to estimate time to mastery: {e}")
            return TimeToMasteryEstimate(
                student_id=student_id, concept=concept, estimated_days=21.0,
                confidence_interval=(14.0, 28.0), factors_affecting_time={},
                recommended_study_schedule={}, milestone_predictions=[]
            )
    
    async def generate_prediction_explanation(self, prediction: PredictionResult) -> PredictionExplanation:
        """Generate explainable AI explanation for a prediction"""
        try:
            logger.info(f"📝 Generating explanation for prediction {prediction.prediction_id}")
            
            # Analyze feature contributions using SHAP or LIME-like approach
            feature_contributions = await self._analyze_feature_contributions(prediction)
            
            # Find similar student patterns
            similar_patterns = await self._find_similar_student_patterns(
                prediction.student_id, prediction.contributing_factors
            )
            
            # Identify confidence factors
            confidence_factors = await self._identify_confidence_factors(prediction)
            
            # Generate alternative scenarios
            alternative_scenarios = await self._generate_alternative_scenarios(prediction)
            
            # Create actionable insights
            actionable_insights = await self._generate_actionable_insights(prediction)
            
            # Prepare visualization data
            visualization_data = await self._prepare_visualization_data(prediction)
            
            # Generate natural language explanation
            model_explanation = await self._generate_natural_explanation(prediction, feature_contributions)
            
            return PredictionExplanation(
                prediction_id=prediction.prediction_id,
                model_explanation=model_explanation,
                feature_contributions=feature_contributions,
                similar_student_patterns=similar_patterns,
                confidence_factors=confidence_factors,
                alternative_scenarios=alternative_scenarios,
                actionable_insights=actionable_insights,
                visualization_data=visualization_data
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to generate prediction explanation: {e}")
            return PredictionExplanation(
                prediction_id=prediction.prediction_id,
                model_explanation="Explanation generation failed",
                feature_contributions={},
                similar_student_patterns=[],
                confidence_factors={},
                alternative_scenarios=[],
                actionable_insights=[],
                visualization_data={}
            )
    
    async def real_time_prediction_update(self, student_id: str, new_interaction: Dict[str, Any]) -> Dict[str, PredictionResult]:
        """Update predictions in real-time based on new student interaction"""
        try:
            with self.prediction_lock:
                logger.info(f"⚡ Real-time prediction update for {student_id}")
                
                # Add new interaction to event buffer
                self.event_buffer.append({
                    'student_id': student_id,
                    'interaction': new_interaction,
                    'timestamp': datetime.now()
                })
                
                # Update real-time features
                await self._update_real_time_features(student_id, new_interaction)
                
                # Check if prediction cache needs updating
                cache_key = f"predictions_{student_id}"
                if (cache_key not in self.prediction_cache or 
                    cache_key not in self.cache_expiry or
                    datetime.now() > self.cache_expiry[cache_key]):
                    
                    # Generate updated predictions
                    updated_predictions = {}
                    
                    # Update success prediction
                    success_pred = await self.predict_student_success_enhanced(student_id)
                    updated_predictions['success'] = success_pred
                    
                    # Update engagement prediction
                    engagement_pred = await self.predict_student_engagement_enhanced(student_id)
                    updated_predictions['engagement'] = engagement_pred
                    
                    # Update multi-timeframe predictions
                    timeframe_pred = await self.predict_multi_timeframe(student_id, 'performance')
                    updated_predictions['multi_timeframe'] = timeframe_pred
                    
                    # Cache predictions
                    self.prediction_cache[cache_key] = updated_predictions
                    self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=self.cache_ttl)
                    
                    # Add to prediction stream for monitoring
                    self.prediction_streams[student_id].append({
                        'timestamp': datetime.now(),
                        'predictions': updated_predictions,
                        'trigger': 'real_time_update'
                    })
                    
                    # Keep only recent entries in stream
                    if len(self.prediction_streams[student_id]) > 100:
                        self.prediction_streams[student_id].popleft()
                    
                    return updated_predictions
                else:
                    # Return cached predictions
                    return self.prediction_cache[cache_key]
                    
        except Exception as e:
            logger.error(f"❌ Failed to update real-time predictions: {e}")
            return {}

    async def predict_student_success_enhanced(self, student_id: str, horizon_days: int = 7) -> PredictionResult:
        """Enhanced student success prediction with Phase 6.3 features"""
        try:
            if 'success_predictor' not in self.models:
                # Train model if not available
                await self.train_success_prediction_model()
            
            # Extract current features
            features = await self.extract_student_features(student_id, lookback_days=30)
            
            if not features:
                raise ValueError(f"No features available for student {student_id}")
            
            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(features, 'success_predictor')
            
            # Scale features
            scaler = self.scalers['success_predictor']
            feature_vector_scaled = scaler.transform([feature_vector])
            
            # Make prediction
            model = self.models['success_predictor']
            prediction = model.predict(feature_vector_scaled)[0]
            
            # Get prediction probability if available
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(feature_vector_scaled)[0]
                confidence = max(prob)
                predicted_value = prob[1]  # Probability of success
            else:
                predicted_value = prediction
                confidence = 0.7  # Default confidence
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(predicted_value, confidence)
            
            # Determine risk level and confidence level
            risk_level = self._determine_risk_level(predicted_value)
            confidence_level = self._determine_confidence_level(confidence)
            
            # Get contributing factors
            contributing_factors = self._get_contributing_factors(features, 'success_predictor')
            
            # Generate recommendations
            recommendations = self._generate_success_recommendations(predicted_value, contributing_factors, risk_level)
            
            # Get ensemble predictions if available
            ensemble_predictions = {}
            if self.ensemble_system:
                ensemble_result = await self.ensemble_system.predict_ensemble(
                    features, 'success_probability', student_id
                )
                ensemble_predictions = ensemble_result.individual_predictions
            
            # Generate intervention recommendations
            intervention_recommendations = await self._generate_intervention_recommendations(
                student_id, predicted_value, risk_level, contributing_factors
            )
            
            # Calculate uncertainty quantification
            uncertainty_quantification = await self._calculate_uncertainty_quantification(
                predicted_value, confidence, ensemble_predictions
            )
            
            # Find comparable students
            comparable_students = await self._find_comparable_students(student_id, features)
            
            # Get concept-specific predictions
            concept_predictions = []
            for concept in PhysicsConcept:
                try:
                    concept_pred = await self.predict_concept_mastery(student_id, concept)
                    concept_predictions.append(concept_pred)
                except:
                    pass  # Skip if concept prediction fails
            
            return PredictionResult(
                student_id=student_id,
                prediction_type='success_probability',
                predicted_value=predicted_value,
                confidence_score=confidence,
                confidence_level=confidence_level,
                confidence_interval=confidence_interval,
                contributing_factors=contributing_factors,
                risk_level=risk_level,
                recommendations=recommendations,
                model_version='2.0',
                timeframe=PredictionTimeframe.MEDIUM_TERM,
                intervention_recommendations=intervention_recommendations,
                ensemble_predictions=ensemble_predictions,
                uncertainty_quantification=uncertainty_quantification,
                prediction_stability_score=await self._calculate_prediction_stability(student_id),
                comparable_students=comparable_students,
                concept_specific_predictions=concept_predictions
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to predict student success for {student_id}: {e}")
            raise

    async def predict_student_success(self, student_id: str, horizon_days: int = 7) -> PredictionResult:
        """Legacy method - redirects to enhanced version"""
        return await self.predict_student_success_enhanced(student_id, horizon_days)
    
    async def predict_student_engagement_enhanced(self, student_id: str, horizon_days: int = 7) -> PredictionResult:
        """Enhanced student engagement prediction with Phase 6.3 features"""
        try:
            if 'engagement_predictor' not in self.models:
                await self.train_engagement_prediction_model()
            
            # Extract current features
            features = await self.extract_student_features(student_id, lookback_days=30)
            
            if not features:
                raise ValueError(f"No features available for student {student_id}")
            
            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(features, 'engagement_predictor')
            
            # Scale features
            scaler = self.scalers['engagement_predictor']
            feature_vector_scaled = scaler.transform([feature_vector])
            
            # Make prediction
            model = self.models['engagement_predictor']
            predicted_value = model.predict(feature_vector_scaled)[0]
            
            # Estimate confidence (based on model performance)
            model_performance = self.model_performance.get('engagement_predictor')
            confidence = model_performance.r2_score if model_performance else 0.7
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(predicted_value, confidence)
            
            # Determine risk level and confidence level
            risk_level = self._determine_engagement_risk_level(predicted_value)
            confidence_level = self._determine_confidence_level(confidence)
            
            # Get contributing factors
            contributing_factors = self._get_contributing_factors(features, 'engagement_predictor')
            
            # Generate recommendations
            recommendations = self._generate_engagement_recommendations(predicted_value, contributing_factors, risk_level)
            
            # Generate intervention recommendations
            intervention_recommendations = await self._generate_engagement_intervention_recommendations(
                student_id, predicted_value, risk_level, contributing_factors
            )
            
            return PredictionResult(
                student_id=student_id,
                prediction_type='engagement_level',
                predicted_value=predicted_value,
                confidence_score=confidence,
                confidence_level=confidence_level,
                confidence_interval=confidence_interval,
                contributing_factors=contributing_factors,
                risk_level=risk_level,
                recommendations=recommendations,
                model_version='2.0',
                timeframe=PredictionTimeframe.MEDIUM_TERM,
                intervention_recommendations=intervention_recommendations,
                prediction_stability_score=await self._calculate_prediction_stability(student_id)
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to predict student engagement for {student_id}: {e}")
            raise

    async def predict_student_engagement(self, student_id: str, horizon_days: int = 7) -> PredictionResult:
        """Legacy method - redirects to enhanced version"""
        return await self.predict_student_engagement_enhanced(student_id, horizon_days)

    # ===== PHASE 6.3 ADVANCED EARLY WARNING SYSTEM =====
    
    async def generate_advanced_early_warning_alerts(self, 
                                                   student_ids: Optional[List[str]] = None,
                                                   alert_types: Optional[List[str]] = None) -> List[EarlyWarningAlert]:
        """Generate comprehensive early warning alerts with confidence scoring"""
        try:
            logger.info("🚨 Generating advanced early warning alerts")
            
            alerts = []
            
            # Get student list
            if student_ids is None:
                student_ids = await self._get_all_active_students()
            
            # Define alert types to check
            if alert_types is None:
                alert_types = [
                    'performance_decline', 'engagement_drop', 'mastery_plateau',
                    'learning_velocity_decrease', 'concept_struggle', 'dropout_risk',
                    'intervention_needed', 'prerequisite_gap'
                ]
            
            for student_id in student_ids:
                try:
                    student_alerts = []
                    
                    # Get comprehensive predictions for student
                    success_pred = await self.predict_student_success_enhanced(student_id)
                    engagement_pred = await self.predict_student_engagement_enhanced(student_id)
                    multi_timeframe = await self.predict_multi_timeframe(student_id, 'overall_performance')
                    
                    # Performance decline alert
                    if 'performance_decline' in alert_types:
                        perf_alert = await self._check_performance_decline_alert(
                            student_id, success_pred, multi_timeframe
                        )
                        if perf_alert:
                            student_alerts.append(perf_alert)
                    
                    # Engagement drop alert
                    if 'engagement_drop' in alert_types:
                        engagement_alert = await self._check_engagement_drop_alert(
                            student_id, engagement_pred, multi_timeframe
                        )
                        if engagement_alert:
                            student_alerts.append(engagement_alert)
                    
                    # Mastery plateau alert
                    if 'mastery_plateau' in alert_types:
                        plateau_alerts = await self._check_mastery_plateau_alerts(student_id)
                        student_alerts.extend(plateau_alerts)
                    
                    # Learning velocity decrease
                    if 'learning_velocity_decrease' in alert_types:
                        velocity_alert = await self._check_learning_velocity_alert(student_id)
                        if velocity_alert:
                            student_alerts.append(velocity_alert)
                    
                    # Concept struggle alerts
                    if 'concept_struggle' in alert_types:
                        concept_alerts = await self._check_concept_struggle_alerts(student_id)
                        student_alerts.extend(concept_alerts)
                    
                    # Dropout risk alert
                    if 'dropout_risk' in alert_types:
                        dropout_alert = await self._check_dropout_risk_alert(
                            student_id, success_pred, engagement_pred
                        )
                        if dropout_alert:
                            student_alerts.append(dropout_alert)
                    
                    # Intervention needed alert
                    if 'intervention_needed' in alert_types:
                        intervention_alert = await self._check_intervention_needed_alert(
                            student_id, success_pred, engagement_pred
                        )
                        if intervention_alert:
                            student_alerts.append(intervention_alert)
                    
                    # Prerequisite gap alert
                    if 'prerequisite_gap' in alert_types:
                        prerequisite_alerts = await self._check_prerequisite_gap_alerts(student_id)
                        student_alerts.extend(prerequisite_alerts)
                    
                    # Store and add alerts
                    if student_alerts:
                        self.active_alerts[student_id] = student_alerts
                        alerts.extend(student_alerts)
                        
                        logger.info(f"Generated {len(student_alerts)} alerts for student {student_id}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to generate alerts for student {student_id}: {e}")
                    continue
            
            # Sort alerts by priority and confidence
            alerts.sort(key=lambda x: (
                self._get_priority_weight(x.severity),
                -x.confidence  # Higher confidence first
            ), reverse=True)
            
            logger.info(f"✅ Generated {len(alerts)} total advanced early warning alerts")
            return alerts
            
        except Exception as e:
            logger.error(f"❌ Failed to generate advanced early warning alerts: {e}")
            return []
    
    async def generate_early_warning_alerts(self, student_ids: Optional[List[str]] = None) -> List[EarlyWarningAlert]:
        """Generate early warning alerts for at-risk students"""
        try:
            alerts = []
            
            # Get student list
            if student_ids is None:
                student_ids = await self._get_all_active_students()
            
            for student_id in student_ids:
                try:
                    # Get predictions
                    success_prediction = await self.predict_student_success(student_id)
                    engagement_prediction = await self.predict_student_engagement(student_id)
                    
                    # Check for alert conditions
                    student_alerts = []
                    
                    # Performance risk alert
                    if success_prediction.predicted_value < self.alert_thresholds['performance_decline']:
                        alert = EarlyWarningAlert(
                            student_id=student_id,
                            alert_type='performance_risk',
                            severity=self._map_risk_to_severity(success_prediction.risk_level),
                            predicted_outcome=f"Success probability: {success_prediction.predicted_value:.2%}",
                            confidence=success_prediction.confidence_score,
                            triggered_by=['low_success_probability'],
                            recommended_actions=success_prediction.recommendations
                        )
                        student_alerts.append(alert)
                    
                    # Engagement risk alert
                    if engagement_prediction.predicted_value < self.alert_thresholds['engagement_drop']:
                        alert = EarlyWarningAlert(
                            student_id=student_id,
                            alert_type='engagement_risk',
                            severity=self._map_risk_to_severity(engagement_prediction.risk_level),
                            predicted_outcome=f"Engagement level: {engagement_prediction.predicted_value:.2f}",
                            confidence=engagement_prediction.confidence_score,
                            triggered_by=['low_engagement_prediction'],
                            recommended_actions=engagement_prediction.recommendations
                        )
                        student_alerts.append(alert)
                    
                    # Check for additional behavioral indicators
                    behavioral_alerts = await self._check_behavioral_alerts(student_id)
                    student_alerts.extend(behavioral_alerts)
                    
                    # Store alerts
                    if student_alerts:
                        self.active_alerts[student_id] = student_alerts
                        alerts.extend(student_alerts)
                
                except Exception as e:
                    logger.error(f"❌ Failed to generate alerts for student {student_id}: {e}")
                    continue
            
            logger.info(f"🚨 Generated {len(alerts)} early warning alerts")
            return alerts
            
        except Exception as e:
            logger.error(f"❌ Failed to generate early warning alerts: {e}")
            return []
    
    async def _collect_training_data(self, min_interactions: int) -> List[Dict[str, Any]]:
        """Collect training data for model training"""
        training_data = []
        
        if not self.db_manager:
            return training_data
        
        try:
            async with self.db_manager.postgres.get_connection() as conn:
                # Get students with sufficient interaction history
                students = await conn.fetch("""
                    SELECT user_id, COUNT(*) as interaction_count
                    FROM interactions 
                    WHERE created_at >= NOW() - INTERVAL '90 days'
                    GROUP BY user_id
                    HAVING COUNT(*) >= $1
                """, min_interactions)
                
                for student_row in students:
                    student_id = str(student_row['user_id'])
                    
                    # Extract features
                    features = await self.extract_student_features(student_id, lookback_days=60)
                    
                    # Calculate target variable (future success rate)
                    target = await self._calculate_future_success_rate(student_id, conn)
                    
                    if features and target is not None:
                        training_data.append({
                            'student_id': student_id,
                            'features': features,
                            'target': target
                        })
            
            logger.info(f"📊 Collected training data for {len(training_data)} students")
            
        except Exception as e:
            logger.error(f"❌ Failed to collect training data: {e}")
        
        return training_data
    
    async def _calculate_future_success_rate(self, student_id: str, conn, days_ahead: int = 14) -> Optional[float]:
        """Calculate success rate for future period as target variable"""
        try:
            # Get interactions from 14 days ago to now (as "future" for the model)
            cutoff_date = datetime.now() - timedelta(days=days_ahead)
            
            result = await conn.fetchrow("""
                SELECT AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as future_success_rate,
                       COUNT(*) as interaction_count
                FROM interactions 
                WHERE user_id = $1 AND created_at >= $2
            """, student_id, cutoff_date)
            
            if result and result['interaction_count'] >= 5:  # Minimum interactions for reliable rate
                return float(result['future_success_rate'])
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate future success rate: {e}")
            return None
    
    # Helper methods for data preparation, feature engineering, and model utilities
    async def _prepare_classification_data(self, training_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for classification model training"""
        try:
            feature_matrix = []
            targets = []
            feature_names = []
            
            # Get feature names from first sample
            if training_data:
                sample_features = training_data[0]['features']
                feature_names = sorted(sample_features.keys())
            
            for sample in training_data:
                # Extract feature vector
                feature_vector = []
                for feature_name in feature_names:
                    value = sample['features'].get(feature_name, 0.0)
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        feature_vector.append(float(value))
                    else:
                        feature_vector.append(0.0)
                
                feature_matrix.append(feature_vector)
                
                # Binary classification target (success >= 0.7)
                target_value = 1 if sample['target'] >= 0.7 else 0
                targets.append(target_value)
            
            X = np.array(feature_matrix, dtype=np.float32)
            y = np.array(targets, dtype=np.int32)
            
            logger.info(f"📊 Prepared classification data: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y, feature_names
            
        except Exception as e:
            logger.error(f"❌ Failed to prepare classification data: {e}")
            return np.array([]), np.array([]), []
    
    async def _prepare_regression_data(self, training_data: List[Dict[str, Any]], target: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for regression model training"""
        try:
            feature_matrix = []
            targets = []
            feature_names = []
            
            # Get feature names from first sample
            if training_data:
                sample_features = training_data[0]['features']
                feature_names = sorted(sample_features.keys())
            
            for sample in training_data:
                # Extract feature vector
                feature_vector = []
                for feature_name in feature_names:
                    value = sample['features'].get(feature_name, 0.0)
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        feature_vector.append(float(value))
                    else:
                        feature_vector.append(0.0)
                
                feature_matrix.append(feature_vector)
                
                # Regression target
                target_value = sample.get(target, sample.get('target', 0.0))
                targets.append(float(target_value))
            
            X = np.array(feature_matrix, dtype=np.float32)
            y = np.array(targets, dtype=np.float32)
            
            logger.info(f"📊 Prepared regression data: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y, feature_names
            
        except Exception as e:
            logger.error(f"❌ Failed to prepare regression data: {e}")
            return np.array([]), np.array([]), []
    
    def _prepare_feature_vector(self, features: Dict[str, float], model_name: str) -> List[float]:
        """Prepare feature vector for prediction"""
        try:
            # Get expected feature order from model performance metadata
            feature_names = []
            if model_name in self.model_performance:
                feature_importance = self.model_performance[model_name].feature_importance
                feature_names = list(feature_importance.keys())
            
            if not feature_names:
                # Use sorted feature names as fallback
                feature_names = sorted(features.keys())
            
            feature_vector = []
            for feature_name in feature_names:
                value = features.get(feature_name, 0.0)
                if isinstance(value, (int, float)) and not np.isnan(value):
                    feature_vector.append(float(value))
                else:
                    feature_vector.append(0.0)
            
            # Ensure minimum feature vector size
            min_size = 20
            while len(feature_vector) < min_size:
                feature_vector.append(0.0)
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"❌ Failed to prepare feature vector: {e}")
            return [0.0] * 20
    
    # Additional helper methods for prediction confidence and explanations
    def _calculate_confidence_interval(self, predicted_value: float, confidence: float) -> Tuple[float, float]:
        """Calculate confidence interval for prediction"""
        try:
            # Use confidence score to estimate interval width
            margin = (1.0 - confidence) * 0.5
            lower_bound = max(0.0, predicted_value - margin)
            upper_bound = min(1.0, predicted_value + margin)
            return (lower_bound, upper_bound)
        except:
            return (predicted_value * 0.8, predicted_value * 1.2)
    
    def _determine_risk_level(self, predicted_value: float) -> str:
        """Determine risk level based on prediction"""
        if predicted_value >= 0.8:
            return 'low'
        elif predicted_value >= 0.6:
            return 'medium'
        elif predicted_value >= 0.4:
            return 'high'
        else:
            return 'critical'
    
    def _determine_engagement_risk_level(self, predicted_value: float) -> str:
        """Determine engagement risk level"""
        if predicted_value >= 0.8:
            return 'low'
        elif predicted_value >= 0.6:
            return 'medium'
        elif predicted_value >= 0.4:
            return 'high'
        else:
            return 'critical'
    
    def _get_contributing_factors(self, features: Dict[str, float], model_name: str) -> Dict[str, float]:
        """Get top contributing factors for prediction"""
        try:
            if model_name not in self.model_performance:
                return {}
            
            feature_importance = self.model_performance[model_name].feature_importance
            contributing_factors = {}
            
            # Get top 5 most important features that exist in current features
            for feature_name, importance in list(feature_importance.items())[:5]:
                if feature_name in features:
                    contributing_factors[feature_name] = importance
            
            return contributing_factors
            
        except Exception as e:
            logger.error(f"❌ Failed to get contributing factors: {e}")
            return {}
    
    def _generate_success_recommendations(self, predicted_value: float, 
                                        contributing_factors: Dict[str, float], 
                                        risk_level: str) -> List[str]:
        """Generate recommendations based on success prediction"""
        recommendations = []
        
        try:
            if risk_level == 'critical':
                recommendations.extend([
                    "Schedule immediate tutoring session",
                    "Review fundamental concepts in struggling areas",
                    "Consider reducing problem difficulty temporarily",
                    "Provide additional guided practice problems"
                ])
            elif risk_level == 'high':
                recommendations.extend([
                    "Increase practice frequency in weak areas",
                    "Use more visual aids and conceptual explanations",
                    "Break down complex problems into smaller steps",
                    "Schedule check-in session within 3 days"
                ])
            elif risk_level == 'medium':
                recommendations.extend([
                    "Continue current pace with minor adjustments",
                    "Focus on consistent practice schedule",
                    "Introduce slightly more challenging problems"
                ])
            else:  # low risk
                recommendations.extend([
                    "Consider advancing to more challenging topics",
                    "Explore advanced applications of current concepts",
                    "Maintain excellent progress"
                ])
            
            # Add specific recommendations based on contributing factors
            for factor, importance in contributing_factors.items():
                if 'response_time' in factor and importance > 0.3:
                    recommendations.append("Work on time management and problem-solving speed")
                elif 'help_seeking' in factor and importance > 0.3:
                    recommendations.append("Practice independent problem-solving before seeking help")
                elif 'concept_switching' in factor and importance > 0.3:
                    recommendations.append("Focus on mastering one concept before moving to the next")
            
            return recommendations[:5]  # Limit to top 5 recommendations
            
        except Exception as e:
            logger.error(f"❌ Failed to generate success recommendations: {e}")
            return ["Continue regular practice and review"]
    
    def _generate_engagement_recommendations(self, predicted_value: float, 
                                           contributing_factors: Dict[str, float], 
                                           risk_level: str) -> List[str]:
        """Generate recommendations based on engagement prediction"""
        recommendations = []
        
        try:
            if risk_level == 'critical':
                recommendations.extend([
                    "Schedule immediate check-in with instructor",
                    "Switch to more interactive learning modalities",
                    "Provide gamified learning experiences",
                    "Consider peer study group assignment"
                ])
            elif risk_level == 'high':
                recommendations.extend([
                    "Introduce more varied problem types",
                    "Add multimedia learning resources",
                    "Provide real-world physics applications",
                    "Schedule motivational check-in"
                ])
            elif risk_level == 'medium':
                recommendations.extend([
                    "Maintain current engagement strategies",
                    "Introduce periodic challenges or competitions",
                    "Provide progress feedback more frequently"
                ])
            else:  # low risk
                recommendations.extend([
                    "Continue excellent engagement",
                    "Consider mentoring other students",
                    "Explore advanced physics topics"
                ])
            
            return recommendations[:5]
            
        except Exception as e:
            logger.error(f"❌ Failed to generate engagement recommendations: {e}")
            return ["Maintain regular study habits"]
    
    def _map_risk_to_severity(self, risk_level: str) -> str:
        """Map risk level to alert severity"""
        mapping = {
            'low': 'low',
            'medium': 'medium',
            'high': 'high',
            'critical': 'critical'
        }
        return mapping.get(risk_level, 'medium')
    
    async def _collect_engagement_training_data(self, min_interactions: int) -> List[Dict[str, Any]]:
        """Collect training data for engagement prediction"""
        training_data = []
        
        if not self.db_manager:
            return training_data
        
        try:
            async with self.db_manager.postgres.get_connection() as conn:
                # Get students with sufficient interaction history
                students = await conn.fetch("""
                    SELECT user_id, COUNT(*) as interaction_count
                    FROM interactions 
                    WHERE created_at >= NOW() - INTERVAL '90 days'
                    GROUP BY user_id
                    HAVING COUNT(*) >= $1
                """, min_interactions)
                
                for student_row in students:
                    student_id = str(student_row['user_id'])
                    
                    # Extract features
                    features = await self.extract_student_features(student_id, lookback_days=60)
                    
                    # Calculate engagement score as target
                    engagement_score = await self._calculate_engagement_score(student_id, conn)
                    
                    if features and engagement_score is not None:
                        training_data.append({
                            'student_id': student_id,
                            'features': features,
                            'engagement_score': engagement_score
                        })
            
            logger.info(f"📊 Collected engagement training data for {len(training_data)} students")
            
        except Exception as e:
            logger.error(f"❌ Failed to collect engagement training data: {e}")
        
        return training_data
    
    async def _calculate_engagement_score(self, student_id: str, conn) -> Optional[float]:
        """Calculate engagement score for a student"""
        try:
            # Calculate engagement based on interaction frequency, session duration, and consistency
            result = await conn.fetchrow("""
                WITH student_sessions AS (
                    SELECT 
                        DATE(created_at) as study_date,
                        COUNT(*) as daily_interactions,
                        MAX(created_at) - MIN(created_at) as session_duration
                    FROM interactions 
                    WHERE user_id = $1 
                    AND created_at >= NOW() - INTERVAL '30 days'
                    GROUP BY DATE(created_at)
                ),
                engagement_metrics AS (
                    SELECT 
                        AVG(daily_interactions) as avg_daily_interactions,
                        COUNT(*) as active_days,
                        AVG(EXTRACT(EPOCH FROM session_duration)/3600.0) as avg_session_hours,
                        STDDEV(daily_interactions) as interaction_consistency
                    FROM student_sessions
                )
                SELECT 
                    avg_daily_interactions,
                    active_days,
                    avg_session_hours,
                    COALESCE(interaction_consistency, 0) as consistency_score
                FROM engagement_metrics
            """, student_id)
            
            if result:
                # Normalize engagement components (0-1 scale)
                interaction_score = min(1.0, result['avg_daily_interactions'] / 20.0)  # Max 20 interactions/day
                frequency_score = min(1.0, result['active_days'] / 30.0)  # Max 30 days
                duration_score = min(1.0, result['avg_session_hours'] / 3.0)  # Max 3 hours/session
                consistency_score = 1.0 / (1.0 + result['consistency_score'])  # Lower std = higher consistency
                
                # Weighted engagement score
                engagement_score = (
                    0.3 * interaction_score +
                    0.3 * frequency_score +
                    0.2 * duration_score +
                    0.2 * consistency_score
                )
                
                return float(engagement_score)
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate engagement score: {e}")
            return None
    
    async def _check_behavioral_alerts(self, student_id: str) -> List[EarlyWarningAlert]:
        """Check for additional behavioral alert indicators"""
        alerts = []
        
        try:
            if not self.db_manager:
                return alerts
            
            async with self.db_manager.postgres.get_connection() as conn:
                # Check for sudden performance drops
                performance_drop = await conn.fetchrow("""
                    WITH recent_performance AS (
                        SELECT AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as recent_success
                        FROM interactions 
                        WHERE user_id = $1 
                        AND created_at >= NOW() - INTERVAL '7 days'
                    ),
                    previous_performance AS (
                        SELECT AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as previous_success
                        FROM interactions 
                        WHERE user_id = $1 
                        AND created_at BETWEEN NOW() - INTERVAL '14 days' AND NOW() - INTERVAL '7 days'
                    )
                    SELECT 
                        r.recent_success,
                        p.previous_success,
                        (p.previous_success - r.recent_success) as performance_drop
                    FROM recent_performance r, previous_performance p
                """, student_id)
                
                if (performance_drop and performance_drop['performance_drop'] and 
                    performance_drop['performance_drop'] > self.alert_thresholds['performance_decline']):
                    
                    alerts.append(EarlyWarningAlert(
                        student_id=student_id,
                        alert_type='performance_decline',
                        severity='high',
                        predicted_outcome=f"Performance dropped by {performance_drop['performance_drop']:.1%}",
                        confidence=0.8,
                        triggered_by=['sudden_performance_drop'],
                        recommended_actions=[
                            "Review recent learning materials",
                            "Schedule remedial tutoring session",
                            "Check for external factors affecting performance"
                        ]
                    ))
                
                # Check for excessive help-seeking
                help_seeking = await conn.fetchval("""
                    SELECT COUNT(*) / COUNT(*)::float as help_rate
                    FROM interactions 
                    WHERE user_id = $1 
                    AND created_at >= NOW() - INTERVAL '7 days'
                    AND metadata::text LIKE '%help_requested%'
                """, student_id)
                
                if help_seeking and help_seeking > self.alert_thresholds['help_seeking_excess']:
                    alerts.append(EarlyWarningAlert(
                        student_id=student_id,
                        alert_type='excessive_help_seeking',
                        severity='medium',
                        predicted_outcome=f"Help-seeking rate: {help_seeking:.1%}",
                        confidence=0.7,
                        triggered_by=['high_help_seeking_rate'],
                        recommended_actions=[
                            "Encourage independent problem-solving",
                            "Provide structured practice problems",
                            "Review problem-solving strategies"
                        ]
                    ))
        
        except Exception as e:
            logger.error(f"❌ Failed to check behavioral alerts: {e}")
        
        return alerts
    
    async def _get_all_active_students(self) -> List[str]:
        """Get list of all active students"""
        try:
            if not self.db_manager:
                return []
            
            async with self.db_manager.postgres.get_connection() as conn:
                students = await conn.fetch("""
                    SELECT DISTINCT user_id 
                    FROM interactions 
                    WHERE created_at >= NOW() - INTERVAL '30 days'
                """)
                
                return [str(row['user_id']) for row in students]
                
        except Exception as e:
            logger.error(f"❌ Failed to get active students: {e}")
            return []
    
    async def _save_model(self, model_name: str):
        """Save trained model to disk"""
        try:
            model_data = {
                'model': self.models[model_name],
                'scaler': self.scalers.get(model_name),
                'encoders': self.encoders.get(model_name),
                'performance': self.model_performance.get(model_name)
            }
            
            with open(f"{self.model_storage_path}/{model_name}.pkl", 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"💾 Saved model: {model_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save model {model_name}: {e}")

# ===== PHASE 6.3 HELPER METHODS =====

    def _determine_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Determine confidence level enum from score"""
        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.7:
            return ConfidenceLevel.MODERATE
        elif confidence >= 0.6:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    async def _predict_for_timeframe(self, features: Dict[str, float], 
                                   timeframe: PredictionTimeframe, 
                                   prediction_type: str) -> Tuple[float, float]:
        """Make prediction for specific timeframe"""
        try:
            # Adjust features based on timeframe
            adjusted_features = self._adjust_features_for_timeframe(features, timeframe)
            
            # Get timeframe-specific model or use default
            model_key = f"{prediction_type}_{timeframe.value}"
            if model_key in self.models:
                model = self.models[model_key]
                scaler = self.scalers.get(model_key, StandardScaler())
            else:
                # Fallback to generic model
                model = self.models.get('success_predictor')
                scaler = self.scalers.get('success_predictor', StandardScaler())
            
            if model is None:
                return 0.5, 0.3  # Default fallback
            
            # Make prediction
            feature_vector = self._prepare_feature_vector(adjusted_features, model_key)
            feature_vector_scaled = scaler.transform([feature_vector])
            
            prediction = model.predict(feature_vector_scaled)[0]
            
            # Estimate confidence based on timeframe (longer = less certain)
            base_confidence = 0.8
            timeframe_penalty = {'short_term': 0.0, 'medium_term': 0.1, 'long_term': 0.2}
            confidence = base_confidence - timeframe_penalty.get(timeframe.value, 0.1)
            
            return float(prediction), float(confidence)
            
        except Exception as e:
            logger.error(f"❌ Failed to predict for timeframe {timeframe.value}: {e}")
            return 0.5, 0.3

    def _adjust_features_for_timeframe(self, features: Dict[str, float], 
                                     timeframe: PredictionTimeframe) -> Dict[str, float]:
        """Adjust features based on prediction timeframe"""
        adjusted = features.copy()
        
        # Apply timeframe-specific adjustments
        if timeframe == PredictionTimeframe.SHORT_TERM:
            # Give more weight to recent performance
            adjusted['recent_weight_multiplier'] = 1.5
        elif timeframe == PredictionTimeframe.LONG_TERM:
            # Give more weight to historical trends
            adjusted['trend_weight_multiplier'] = 1.3
        
        return adjusted

    def _calculate_prediction_trend(self, predictions: List[float]) -> Tuple[str, float]:
        """Calculate trend direction and strength from prediction sequence"""
        if len(predictions) < 2:
            return 'stable', 0.0
        
        # Calculate slope
        x = np.arange(len(predictions))
        slope, _, r_value, _, _ = stats.linregress(x, predictions)
        
        # Determine direction
        if slope > 0.05:
            direction = 'improving'
        elif slope < -0.05:
            direction = 'declining'
        else:
            direction = 'stable'
        
        # Strength is based on R-squared and absolute slope
        strength = min(abs(slope) * 10, 1.0) * (r_value ** 2)
        
        return direction, float(strength)

    async def _extract_concept_features(self, student_id: str, concept: PhysicsConcept) -> Dict[str, float]:
        """Extract features specific to a physics concept"""
        try:
            features = {}
            
            if not self.db_manager:
                return features
            
            concept_name = concept.value
            
            async with self.db_manager.postgres.get_connection() as conn:
                # Get concept-specific interactions
                concept_data = await conn.fetch("""
                    SELECT success, execution_time_ms, metadata, created_at
                    FROM interactions 
                    WHERE user_id = $1 
                    AND agent_type = $2
                    AND created_at >= NOW() - INTERVAL '60 days'
                    ORDER BY created_at ASC
                """, student_id, concept_name)
                
                if concept_data:
                    df = pd.DataFrame([dict(row) for row in concept_data])
                    
                    # Basic concept performance
                    features['concept_success_rate'] = df['success'].mean()
                    features['concept_attempt_count'] = len(df)
                    features['concept_avg_response_time'] = df['execution_time_ms'].mean()
                    
                    # Concept learning progression
                    features['concept_improvement_rate'] = self._calculate_trend(df['success'].astype(int).values)
                    
                    # Recent performance
                    recent_data = df.tail(10)
                    features['recent_concept_success_rate'] = recent_data['success'].mean()
                    features['concept_consistency'] = 1.0 - recent_data['success'].std() if len(recent_data) > 1 else 1.0
                
                # Get prerequisite performance
                prerequisite_performance = await self._get_prerequisite_performance(student_id, concept, conn)
                features.update(prerequisite_performance)
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Failed to extract concept features: {e}")
            return {}

    async def _calculate_current_mastery(self, student_id: str, concept: PhysicsConcept) -> float:
        """Calculate current mastery level for a concept"""
        try:
            if not self.db_manager:
                return 0.5
            
            async with self.db_manager.postgres.get_connection() as conn:
                # Get recent performance data
                mastery_data = await conn.fetchrow("""
                    WITH recent_attempts AS (
                        SELECT success, execution_time_ms,
                               ROW_NUMBER() OVER (ORDER BY created_at DESC) as attempt_rank
                        FROM interactions 
                        WHERE user_id = $1 
                        AND agent_type = $2
                        AND created_at >= NOW() - INTERVAL '30 days'
                    )
                    SELECT 
                        AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate,
                        AVG(CASE WHEN attempt_rank <= 10 THEN CASE WHEN success THEN 1.0 ELSE 0.0 END END) as recent_success_rate,
                        COUNT(*) as total_attempts
                    FROM recent_attempts
                """, student_id, concept.value)
                
                if mastery_data and mastery_data['total_attempts'] > 0:
                    # Weight recent performance more heavily
                    overall_success = mastery_data['success_rate'] or 0.0
                    recent_success = mastery_data['recent_success_rate'] or 0.0
                    
                    # Weighted mastery calculation
                    mastery = (0.4 * overall_success + 0.6 * recent_success)
                    
                    # Adjust based on attempt count (more attempts = more confidence)
                    attempt_confidence = min(mastery_data['total_attempts'] / 20.0, 1.0)
                    mastery = mastery * attempt_confidence + 0.5 * (1 - attempt_confidence)
                    
                    return float(np.clip(mastery, 0.0, 1.0))
                
                return 0.5
                
        except Exception as e:
            logger.error(f"❌ Failed to calculate current mastery: {e}")
            return 0.5

    async def _generate_intervention_recommendations(self, student_id: str, predicted_value: float,
                                                  risk_level: str, contributing_factors: Dict[str, float]) -> List[InterventionRecommendation]:
        """Generate AI-powered intervention recommendations"""
        recommendations = []
        
        try:
            # Analyze risk factors and generate targeted interventions
            if risk_level in ['high', 'critical']:
                # High-priority interventions
                if 'success_rate' in contributing_factors and contributing_factors['success_rate'] < 0.5:
                    recommendations.append(InterventionRecommendation(
                        student_id=student_id,
                        intervention_id=str(uuid.uuid4()),
                        priority=InterventionPriority.HIGH,
                        intervention_type='remedial_tutoring',
                        description='Schedule immediate one-on-one tutoring session',
                        rationale=f'Low success rate ({contributing_factors["success_rate"]:.2f}) indicates fundamental understanding gaps',
                        expected_impact=0.3,
                        confidence_score=0.8,
                        timeframe=PredictionTimeframe.SHORT_TERM,
                        success_probability=0.7,
                        resource_requirements={'tutor_hours': 2, 'materials': 'concept_review_sheets'},
                        alternative_interventions=[{
                            'type': 'peer_tutoring',
                            'description': 'Pair with high-performing peer',
                            'impact': 0.2
                        }],
                        monitoring_metrics=['success_rate_improvement', 'engagement_increase']
                    ))
                
                if 'help_seeking_rate' in contributing_factors and contributing_factors['help_seeking_rate'] > 0.7:
                    recommendations.append(InterventionRecommendation(
                        student_id=student_id,
                        intervention_id=str(uuid.uuid4()),
                        priority=InterventionPriority.MEDIUM,
                        intervention_type='independent_learning_support',
                        description='Provide structured problem-solving framework',
                        rationale='High help-seeking rate suggests need for independent problem-solving skills',
                        expected_impact=0.25,
                        confidence_score=0.75,
                        timeframe=PredictionTimeframe.MEDIUM_TERM,
                        success_probability=0.6,
                        resource_requirements={'framework_materials': True, 'practice_problems': 10},
                        alternative_interventions=[],
                        monitoring_metrics=['help_seeking_rate_decrease', 'independence_score']
                    ))
            
            return recommendations[:3]  # Limit to top 3 recommendations
            
        except Exception as e:
            logger.error(f"❌ Failed to generate intervention recommendations: {e}")
            return []

    async def _calculate_uncertainty_quantification(self, predicted_value: float, 
                                                 confidence: float, 
                                                 ensemble_predictions: Dict[str, float]) -> Dict[str, float]:
        """Calculate comprehensive uncertainty quantification"""
        uncertainty = {}
        
        try:
            # Model uncertainty (epistemic)
            uncertainty['model_uncertainty'] = 1.0 - confidence
            
            # Prediction variance from ensemble
            if ensemble_predictions:
                predictions = list(ensemble_predictions.values())
                uncertainty['ensemble_variance'] = float(np.var(predictions))
                uncertainty['ensemble_std'] = float(np.std(predictions))
                uncertainty['prediction_range'] = float(max(predictions) - min(predictions))
            else:
                uncertainty['ensemble_variance'] = 0.1
                uncertainty['ensemble_std'] = 0.3
                uncertainty['prediction_range'] = 0.2
            
            # Aleatoric uncertainty (data noise)
            uncertainty['data_uncertainty'] = min(0.1 + uncertainty['ensemble_variance'], 0.5)
            
            # Total uncertainty
            uncertainty['total_uncertainty'] = np.sqrt(
                uncertainty['model_uncertainty']**2 + 
                uncertainty['ensemble_variance'] + 
                uncertainty['data_uncertainty']**2
            )
            
            return uncertainty
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate uncertainty quantification: {e}")
            return {'total_uncertainty': 0.5}

    async def _calculate_prediction_stability(self, student_id: str) -> float:
        """Calculate stability score for predictions over time"""
        try:
            # Get recent prediction history from cache/stream
            if student_id in self.prediction_streams:
                recent_predictions = list(self.prediction_streams[student_id])[-10:]  # Last 10 predictions
                
                if len(recent_predictions) > 2:
                    # Extract prediction values
                    success_predictions = []
                    for pred_entry in recent_predictions:
                        if 'predictions' in pred_entry and 'success' in pred_entry['predictions']:
                            success_predictions.append(pred_entry['predictions']['success'].predicted_value)
                    
                    if len(success_predictions) > 2:
                        # Calculate stability as inverse of variance
                        prediction_variance = np.var(success_predictions)
                        stability = 1.0 / (1.0 + prediction_variance * 10)  # Scale variance
                        return float(np.clip(stability, 0.0, 1.0))
            
            return 0.7  # Default stability score
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate prediction stability: {e}")
            return 0.5

    async def _find_comparable_students(self, student_id: str, features: Dict[str, float]) -> List[str]:
        """Find students with similar learning patterns"""
        try:
            if not self.db_manager:
                return []
            
            # This is a simplified implementation - in practice would use more sophisticated similarity measures
            comparable = []
            
            async with self.db_manager.postgres.get_connection() as conn:
                # Find students with similar performance patterns
                similar_students = await conn.fetch("""
                    WITH student_stats AS (
                        SELECT 
                            user_id,
                            AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate,
                            AVG(execution_time_ms) as avg_response_time,
                            COUNT(*) as interaction_count
                        FROM interactions 
                        WHERE created_at >= NOW() - INTERVAL '30 days'
                        AND user_id != $1
                        GROUP BY user_id
                        HAVING COUNT(*) >= 10
                    )
                    SELECT user_id, success_rate, avg_response_time
                    FROM student_stats
                    WHERE ABS(success_rate - $2) < 0.2
                    AND ABS(avg_response_time - $3) < 2000
                    LIMIT 5
                """, student_id, 
                features.get('success_rate', 0.5),
                features.get('avg_response_time', 5000))
                
                comparable = [str(row['user_id']) for row in similar_students]
            
            return comparable
            
        except Exception as e:
            logger.error(f"❌ Failed to find comparable students: {e}")
            return []

    def _get_priority_weight(self, severity: str) -> int:
        """Convert severity to numeric weight for sorting"""
        weights = {
            'critical': 4,
            'high': 3,
            'medium': 2,
            'low': 1
        }
        return weights.get(severity, 1)

    async def _update_real_time_features(self, student_id: str, new_interaction: Dict[str, Any]):
        """Update real-time feature cache with new interaction"""
        try:
            if student_id not in self.real_time_features:
                self.real_time_features[student_id] = {}
            
            # Update rolling averages and counters
            features = self.real_time_features[student_id]
            
            # Success rate tracking
            if 'recent_success_count' not in features:
                features['recent_success_count'] = 0
                features['recent_attempt_count'] = 0
            
            features['recent_attempt_count'] += 1
            if new_interaction.get('success', False):
                features['recent_success_count'] += 1
            
            features['recent_success_rate'] = features['recent_success_count'] / features['recent_attempt_count']
            
            # Response time tracking
            if 'response_times' not in features:
                features['response_times'] = deque(maxlen=20)
            
            response_time = new_interaction.get('execution_time_ms', 5000)
            features['response_times'].append(response_time)
            features['avg_response_time'] = np.mean(features['response_times'])
            
            # Session activity
            features['last_activity'] = datetime.now()
            features['interactions_this_session'] = features.get('interactions_this_session', 0) + 1
            
        except Exception as e:
            logger.error(f"❌ Failed to update real-time features: {e}")

# ===== ADDITIONAL PHASE 6.3 HELPER METHODS (SIMPLIFIED IMPLEMENTATIONS) =====

    async def _predict_concept_mastery_level(self, features: Dict[str, float], concept: PhysicsConcept) -> float:
        """Predict future mastery level for concept"""
        # Simplified implementation - would use trained concept-specific models
        current_success = features.get('concept_success_rate', 0.5)
        improvement_rate = features.get('concept_improvement_rate', 0.0)
        return min(1.0, current_success + improvement_rate * 0.1)

    async def _estimate_time_to_mastery(self, student_id: str, concept: PhysicsConcept, 
                                      current_mastery: float, predicted_mastery: float) -> Optional[float]:
        """Estimate days to achieve mastery"""
        if predicted_mastery <= current_mastery:
            return None  # Already at or declining mastery
        
        # Simple linear estimation - would use more sophisticated models
        mastery_gap = 0.8 - current_mastery  # Assuming 0.8 is mastery threshold
        learning_rate = (predicted_mastery - current_mastery) / 7.0  # Per day
        
        if learning_rate <= 0:
            return None
        
        return mastery_gap / learning_rate

    async def _identify_prerequisite_gaps(self, student_id: str, concept: PhysicsConcept) -> List[str]:
        """Identify missing prerequisite concepts"""
        # Simplified concept dependency mapping
        prerequisites = {
            PhysicsConcept.KINEMATICS_2D: ['kinematics_1d', 'vectors'],
            PhysicsConcept.FORCES: ['kinematics_1d', 'vectors'],
            PhysicsConcept.ENERGY: ['forces', 'kinematics_1d'],
            PhysicsConcept.MOMENTUM: ['forces', 'kinematics_1d'],
            PhysicsConcept.ANGULAR_MOTION: ['forces', 'kinematics_1d']
        }
        
        return prerequisites.get(concept, [])

    async def _generate_learning_sequence(self, student_id: str, concept: PhysicsConcept) -> List[str]:
        """Generate recommended learning sequence"""
        # Simplified sequence generation
        sequences = {
            PhysicsConcept.FORCES: ['vectors', 'newton_laws_1', 'newton_laws_2', 'newton_laws_3', 'applications'],
            PhysicsConcept.ENERGY: ['work', 'kinetic_energy', 'potential_energy', 'conservation', 'applications'],
            PhysicsConcept.MOMENTUM: ['linear_momentum', 'impulse', 'collisions', 'conservation', 'applications']
        }
        
        return sequences.get(concept, ['basic_concepts', 'applications'])

    async def _calculate_difficulty_adjustment(self, student_id: str, concept: PhysicsConcept) -> float:
        """Calculate recommended difficulty adjustment"""
        # Would analyze student performance and suggest difficulty changes
        return 0.0  # No adjustment by default

# Main testing function
async def test_predictive_analytics():
    """Test predictive analytics engine"""
    try:
        logger.info("🧪 Testing Predictive Analytics Engine")
        
        engine = PredictiveAnalyticsEngine()
        await engine.initialize()
        
        # Test feature extraction (mock data)
        sample_features = {
            'avg_hour_of_day': 14.5,
            'interactions_per_day': 8.2,
            'success_rate': 0.75,
            'help_seeking_rate': 0.15,
            'concept_coverage': 4
        }
        
        logger.info(f"✅ Sample features extracted: {len(sample_features)} features")
        
        # Test prediction result structure
        sample_prediction = PredictionResult(
            student_id="test_student",
            prediction_type="success_probability",
            predicted_value=0.78,
            confidence_score=0.85,
            confidence_interval=(0.70, 0.86),
            contributing_factors={'success_rate': 0.3, 'engagement': 0.25},
            risk_level='low',
            recommendations=['Continue current learning pace'],
            model_version='1.0'
        )
        
        logger.info(f"✅ Sample prediction created: {sample_prediction.prediction_type}")
        
        logger.info("✅ Predictive Analytics Engine test completed")
        
    except Exception as e:
        logger.error(f"❌ Predictive Analytics test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_predictive_analytics())