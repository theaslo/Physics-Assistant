#!/usr/bin/env python3
"""
Advanced Early Warning System for Physics Assistant Phase 6
Implements sophisticated predictive models for early intervention,
risk assessment, and personalized support recommendations.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
import lightgbm as lgb
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from enum import Enum
import pickle
import warnings
import redis
import hashlib

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class InterventionType(Enum):
    ACADEMIC_SUPPORT = "academic_support"
    MOTIVATIONAL = "motivational"
    TECHNICAL_HELP = "technical_help"
    LEARNING_STRATEGY = "learning_strategy"
    SOCIAL_SUPPORT = "social_support"
    DIFFICULTY_ADJUSTMENT = "difficulty_adjustment"

@dataclass
class RiskIndicator:
    """Individual risk indicator with prediction confidence"""
    indicator_name: str
    risk_score: float          # 0-1 scale
    confidence: float          # Model confidence
    trend_direction: str       # 'improving', 'stable', 'declining'
    severity: RiskLevel
    contributing_factors: Dict[str, float]
    time_horizon: int          # Days ahead prediction
    reliability_score: float   # Historical accuracy of this indicator

@dataclass
class StudentRiskProfile:
    """Comprehensive risk assessment for a student"""
    student_id: str
    overall_risk_score: float
    risk_level: RiskLevel
    risk_indicators: List[RiskIndicator]
    predicted_outcomes: Dict[str, float]
    intervention_recommendations: List['InterventionRecommendation']
    confidence_bounds: Tuple[float, float]
    assessment_timestamp: datetime = field(default_factory=datetime.now)
    next_assessment_due: datetime = field(default_factory=lambda: datetime.now() + timedelta(days=1))

@dataclass
class InterventionRecommendation:
    """Personalized intervention recommendation"""
    intervention_id: str
    intervention_type: InterventionType
    priority_level: int        # 1-5 scale
    description: str
    rationale: str
    expected_impact: float     # Expected improvement
    urgency_score: float       # How urgent is this intervention
    effort_required: str       # 'low', 'medium', 'high'
    success_probability: float
    implementation_steps: List[str]
    monitoring_metrics: List[str]

@dataclass
class EarlyWarningAlert:
    """Enhanced early warning alert"""
    alert_id: str
    student_id: str
    alert_type: str
    severity: RiskLevel
    risk_indicators: List[str]
    predicted_outcome: str
    confidence: float
    intervention_recommendations: List[InterventionRecommendation]
    alert_triggers: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(days=7))
    acknowledged: bool = False
    resolved: bool = False

class DeepRiskPredictor(nn.Module):
    """Deep neural network for comprehensive risk prediction"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64], 
                 num_risk_types: int = 6, dropout_rate: float = 0.3):
        super(DeepRiskPredictor, self).__init__()
        
        # Shared feature extractor
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Multiple prediction heads for different risk types
        self.overall_risk_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.performance_risk_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.engagement_risk_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.dropout_risk_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_risk_types),
            nn.Softplus()
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        
        overall_risk = self.overall_risk_head(features)
        performance_risk = self.performance_risk_head(features)
        engagement_risk = self.engagement_risk_head(features)
        dropout_risk = self.dropout_risk_head(features)
        
        uncertainties = self.uncertainty_head(features)
        
        return {
            'overall_risk': overall_risk,
            'performance_risk': performance_risk,
            'engagement_risk': engagement_risk,
            'dropout_risk': dropout_risk,
            'uncertainties': uncertainties
        }

class AnomalyDetector:
    """Anomaly detection for unusual student behavior patterns"""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Behavioral baselines
        self.baseline_patterns = {}
        self.student_profiles = {}
    
    def fit(self, student_features: pd.DataFrame):
        """Fit anomaly detection models on student data"""
        try:
            # Scale features
            scaled_features = self.scaler.fit_transform(student_features)
            
            # Reduce dimensionality
            reduced_features = self.pca.fit_transform(scaled_features)
            
            # Fit anomaly detectors
            self.isolation_forest.fit(reduced_features)
            self.dbscan.fit(reduced_features)
            
            # Establish behavioral baselines
            for student_id in student_features.index:
                student_data = student_features.loc[student_id]
                self.baseline_patterns[student_id] = {
                    'mean_values': student_data.mean(),
                    'std_values': student_data.std(),
                    'percentiles': student_data.quantile([0.25, 0.5, 0.75])
                }
            
            self.is_fitted = True
            logger.info("✅ Anomaly detection models fitted successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to fit anomaly detection models: {e}")
    
    def detect_anomalies(self, student_id: str, current_features: np.ndarray) -> Dict[str, Any]:
        """Detect anomalies in student behavior"""
        try:
            if not self.is_fitted:
                return {'anomaly_score': 0.0, 'is_anomaly': False, 'anomaly_types': []}
            
            # Scale and transform features
            scaled_features = self.scaler.transform([current_features])
            reduced_features = self.pca.transform(scaled_features)
            
            # Isolation Forest anomaly score
            isolation_score = self.isolation_forest.decision_function(reduced_features)[0]
            is_anomaly_isolation = self.isolation_forest.predict(reduced_features)[0] == -1
            
            # Check against baseline patterns
            baseline_anomalies = []
            if student_id in self.baseline_patterns:
                baseline = self.baseline_patterns[student_id]
                current_df = pd.Series(current_features)
                
                # Check for significant deviations
                for i, (mean_val, std_val) in enumerate(zip(baseline['mean_values'], baseline['std_values'])):
                    if std_val > 0:
                        z_score = abs((current_features[i] - mean_val) / std_val)
                        if z_score > 3:  # More than 3 standard deviations
                            baseline_anomalies.append(f'feature_{i}_deviation')
            
            # Combine anomaly indicators
            anomaly_score = max(0.0, min(1.0, (1.0 - isolation_score) / 2.0))
            is_anomaly = is_anomaly_isolation or len(baseline_anomalies) > 0
            
            return {
                'anomaly_score': anomaly_score,
                'is_anomaly': is_anomaly,
                'anomaly_types': baseline_anomalies,
                'isolation_score': isolation_score
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to detect anomalies: {e}")
            return {'anomaly_score': 0.0, 'is_anomaly': False, 'anomaly_types': []}

class AdvancedEarlyWarningSystem:
    """Advanced early warning system with multiple ML models and intervention strategies"""
    
    def __init__(self, db_manager=None, redis_client=None, model_storage_path="/tmp/ml_models"):
        self.db_manager = db_manager
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, db=0)
        self.model_storage_path = model_storage_path
        
        # ML Models
        self.deep_risk_predictor = None
        self.anomaly_detector = AnomalyDetector()
        self.ensemble_models = {}
        
        # Risk assessment components
        self.risk_indicators = {}
        self.intervention_engine = None
        self.alert_manager = AlertManager()
        
        # Historical data for trend analysis
        self.risk_history = defaultdict(list)
        self.intervention_outcomes = defaultdict(list)
        
        # Configuration
        self.config = {
            'risk_thresholds': {
                'low': 0.3,
                'medium': 0.6,
                'high': 0.8,
                'critical': 0.95
            },
            'prediction_horizons': [1, 3, 7, 14],  # Days ahead
            'min_data_points': 10,
            'update_frequency': 3600,  # Seconds
            'intervention_cooldown': 86400,  # 24 hours
            'alert_retention_days': 30
        }
        
        # Feature importance tracking
        self.feature_importance_history = defaultdict(list)
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_queue = deque(maxlen=1000)
    
    async def initialize(self):
        """Initialize the advanced early warning system"""
        try:
            logger.info("🚀 Initializing Advanced Early Warning System")
            
            # Create storage directories
            import os
            os.makedirs(self.model_storage_path, exist_ok=True)
            
            # Initialize components
            await self._initialize_ml_models()
            await self._initialize_risk_indicators()
            await self._initialize_intervention_engine()
            
            # Load historical data
            await self._load_historical_data()
            
            # Start monitoring if database available
            if self.db_manager:
                await self._start_real_time_monitoring()
            
            logger.info("✅ Advanced Early Warning System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Advanced Early Warning System: {e}")
            return False
    
    async def _initialize_ml_models(self):
        """Initialize machine learning models"""
        try:
            # Estimate feature dimensions
            feature_dim = 75  # Comprehensive feature set
            
            # Initialize deep neural network
            self.deep_risk_predictor = DeepRiskPredictor(
                input_dim=feature_dim,
                hidden_dims=[256, 128, 64],
                num_risk_types=6
            )
            
            # Initialize ensemble models
            self.ensemble_models = {
                'xgboost_classifier': xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                ),
                'lightgbm_classifier': lgb.LGBMClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1
                ),
                'random_forest': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=42
                )
            }
            
            logger.info("✅ ML models initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ML models: {e}")
    
    async def _initialize_risk_indicators(self):
        """Initialize risk indicator calculation methods"""
        try:
            self.risk_indicators = {
                'performance_decline': self._calculate_performance_decline_risk,
                'engagement_drop': self._calculate_engagement_drop_risk,
                'learning_plateau': self._calculate_learning_plateau_risk,
                'help_seeking_excess': self._calculate_help_seeking_risk,
                'time_management_issues': self._calculate_time_management_risk,
                'concept_confusion': self._calculate_concept_confusion_risk,
                'social_isolation': self._calculate_social_isolation_risk,
                'technical_difficulties': self._calculate_technical_difficulties_risk
            }
            
            logger.info("✅ Risk indicators initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize risk indicators: {e}")
    
    async def _initialize_intervention_engine(self):
        """Initialize intervention recommendation engine"""
        try:
            self.intervention_engine = InterventionEngine()
            await self.intervention_engine.initialize()
            
            logger.info("✅ Intervention engine initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize intervention engine: {e}")
    
    async def _load_historical_data(self):
        """Load historical risk and intervention data"""
        try:
            if not self.db_manager:
                return
            
            # Load risk assessment history
            async with self.db_manager.postgres.get_connection() as conn:
                risk_history = await conn.fetch("""
                    SELECT student_id, risk_type, risk_score, assessment_date
                    FROM risk_assessments 
                    WHERE assessment_date >= $1
                    ORDER BY assessment_date DESC
                """, datetime.now() - timedelta(days=30))
                
                for record in risk_history:
                    student_id = str(record['student_id'])
                    self.risk_history[student_id].append({
                        'risk_type': record['risk_type'],
                        'risk_score': record['risk_score'],
                        'date': record['assessment_date']
                    })
            
            logger.info(f"📊 Loaded risk history for {len(self.risk_history)} students")
            
        except Exception as e:
            logger.error(f"❌ Failed to load historical data: {e}")
    
    async def _start_real_time_monitoring(self):
        """Start real-time monitoring of student activities"""
        try:
            self.monitoring_active = True
            # In a real implementation, this would set up event listeners
            logger.info("🔍 Real-time monitoring started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start real-time monitoring: {e}")
    
    async def assess_student_risk(self, student_id: str, 
                                prediction_horizon: int = 7) -> StudentRiskProfile:
        """Perform comprehensive risk assessment for a student"""
        try:
            logger.info(f"🔍 Assessing risk for student {student_id}")
            
            # Extract comprehensive features
            features = await self._extract_risk_features(student_id)
            
            if not features:
                return self._create_default_risk_profile(student_id)
            
            # Calculate individual risk indicators
            risk_indicators = []
            for indicator_name, calculator in self.risk_indicators.items():
                indicator = await calculator(student_id, features, prediction_horizon)
                if indicator:
                    risk_indicators.append(indicator)
            
            # Run ensemble predictions
            ensemble_predictions = await self._run_ensemble_predictions(features, student_id)
            
            # Calculate overall risk score
            overall_risk_score = await self._calculate_overall_risk(
                risk_indicators, ensemble_predictions
            )
            
            # Determine risk level
            risk_level = self._determine_risk_level(overall_risk_score)
            
            # Generate intervention recommendations
            interventions = await self.intervention_engine.generate_recommendations(
                student_id, risk_indicators, overall_risk_score
            )
            
            # Calculate confidence bounds
            confidence_bounds = self._calculate_confidence_bounds(
                ensemble_predictions, risk_indicators
            )
            
            # Predict specific outcomes
            predicted_outcomes = await self._predict_specific_outcomes(
                student_id, features, prediction_horizon
            )
            
            # Create risk profile
            risk_profile = StudentRiskProfile(
                student_id=student_id,
                overall_risk_score=overall_risk_score,
                risk_level=risk_level,
                risk_indicators=risk_indicators,
                predicted_outcomes=predicted_outcomes,
                intervention_recommendations=interventions,
                confidence_bounds=confidence_bounds
            )
            
            # Store assessment results
            await self._store_risk_assessment(risk_profile)
            
            # Generate alerts if necessary
            await self._check_and_generate_alerts(risk_profile)
            
            logger.info(f"✅ Risk assessment completed: {risk_level.value} risk ({overall_risk_score:.2f})")
            return risk_profile
            
        except Exception as e:
            logger.error(f"❌ Failed to assess student risk: {e}")
            return self._create_default_risk_profile(student_id)
    
    async def _extract_risk_features(self, student_id: str) -> Dict[str, Any]:
        """Extract comprehensive features for risk assessment"""
        try:
            features = {
                'behavioral_features': {},
                'academic_features': {},
                'engagement_features': {},
                'temporal_features': {},
                'social_features': {},
                'technical_features': {}
            }
            
            if not self.db_manager:
                return features
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            async with self.db_manager.postgres.get_connection() as conn:
                # Get interaction data
                interactions = await conn.fetch("""
                    SELECT * FROM interactions 
                    WHERE user_id = $1 AND created_at BETWEEN $2 AND $3
                    ORDER BY created_at ASC
                """, student_id, start_date, end_date)
                
                if not interactions:
                    return features
                
                df = pd.DataFrame([dict(row) for row in interactions])
                df['created_at'] = pd.to_datetime(df['created_at'])
                
                # Extract behavioral features
                features['behavioral_features'] = self._extract_behavioral_risk_features(df)
                
                # Extract academic performance features
                features['academic_features'] = self._extract_academic_risk_features(df, student_id, conn)
                
                # Extract engagement features
                features['engagement_features'] = self._extract_engagement_risk_features(df)
                
                # Extract temporal patterns
                features['temporal_features'] = self._extract_temporal_risk_features(df)
                
                # Extract social interaction features
                features['social_features'] = await self._extract_social_risk_features(student_id, conn)
                
                # Extract technical/system features
                features['technical_features'] = self._extract_technical_risk_features(df)
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Failed to extract risk features for {student_id}: {e}")
            return {}
    
    def _extract_behavioral_risk_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract behavioral risk indicators"""
        features = {}
        
        try:
            total_interactions = len(df)
            if total_interactions == 0:
                return features
            
            # Session patterns
            df['time_diff'] = df['created_at'].diff().dt.total_seconds() / 60
            session_breaks = df['time_diff'] > 30
            
            sessions = df.groupby(session_breaks.cumsum())
            features['avg_session_duration'] = sessions['time_diff'].sum().mean()
            features['session_consistency'] = 1.0 - (sessions.size().std() / sessions.size().mean()) if sessions.size().std() > 0 else 1.0
            
            # Help-seeking patterns
            help_requests = 0
            rapid_submissions = 0
            
            for _, row in df.iterrows():
                if row.get('metadata'):
                    try:
                        metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
                        if metadata.get('help_requested'):
                            help_requests += 1
                    except:
                        pass
                
                # Detect rapid submissions (< 10 seconds)
                if row.get('execution_time_ms', 0) < 10000:
                    rapid_submissions += 1
            
            features['help_seeking_rate'] = help_requests / total_interactions
            features['rapid_submission_rate'] = rapid_submissions / total_interactions
            
            # Response time patterns
            if 'execution_time_ms' in df.columns:
                response_times = df['execution_time_ms'] / 1000.0  # Convert to seconds
                features['avg_response_time'] = response_times.mean()
                features['response_time_variability'] = response_times.std()
                
                # Detect concerning patterns
                very_fast_responses = (response_times < 5).sum()  # Less than 5 seconds
                very_slow_responses = (response_times > 300).sum()  # More than 5 minutes
                
                features['impulsive_response_rate'] = very_fast_responses / total_interactions
                features['struggling_response_rate'] = very_slow_responses / total_interactions
            
            # Concept switching behavior
            if 'agent_type' in df.columns:
                concept_switches = (df['agent_type'] != df['agent_type'].shift()).sum() - 1
                features['concept_switching_rate'] = concept_switches / total_interactions
                
                # Calculate focus persistence
                concept_runs = df.groupby((df['agent_type'] != df['agent_type'].shift()).cumsum()).size()
                features['avg_concept_persistence'] = concept_runs.mean()
            
        except Exception as e:
            logger.error(f"❌ Failed to extract behavioral risk features: {e}")
        
        return features
    
    def _extract_academic_risk_features(self, df: pd.DataFrame, student_id: str, conn) -> Dict[str, float]:
        """Extract academic performance risk indicators"""
        features = {}
        
        try:
            if 'success' in df.columns and len(df) > 0:
                # Performance trends
                success_values = df['success'].astype(int)
                features['current_success_rate'] = success_values.mean()
                
                # Recent vs earlier performance
                if len(success_values) >= 10:
                    recent_performance = success_values.tail(5).mean()
                    earlier_performance = success_values.head(5).mean()
                    features['performance_trend'] = recent_performance - earlier_performance
                
                # Performance volatility
                if len(success_values) >= 5:
                    rolling_success = success_values.rolling(window=5).mean()
                    features['performance_volatility'] = rolling_success.std()
                
                # Difficulty adaptation
                if 'agent_type' in df.columns:
                    difficulty_map = {'math': 1, 'kinematics': 2, 'forces': 3, 'energy': 4, 'momentum': 4, 'angular_motion': 5}
                    df['difficulty'] = df['agent_type'].map(difficulty_map).fillna(3)
                    
                    # Success vs difficulty correlation
                    if df['difficulty'].std() > 0:
                        correlation = df['success'].corr(df['difficulty'])
                        features['difficulty_success_correlation'] = correlation if not np.isnan(correlation) else 0.0
                    
                    # Check for difficulty avoidance
                    high_difficulty_attempts = (df['difficulty'] >= 4).sum()
                    features['high_difficulty_avoidance'] = 1.0 - (high_difficulty_attempts / len(df))
                
                # Error patterns
                error_sequences = []
                current_errors = 0
                for success in success_values:
                    if success == 0:
                        current_errors += 1
                    else:
                        if current_errors > 0:
                            error_sequences.append(current_errors)
                        current_errors = 0
                
                if error_sequences:
                    features['max_consecutive_errors'] = max(error_sequences)
                    features['avg_error_sequence_length'] = np.mean(error_sequences)
                else:
                    features['max_consecutive_errors'] = 0
                    features['avg_error_sequence_length'] = 0
        
        except Exception as e:
            logger.error(f"❌ Failed to extract academic risk features: {e}")
        
        return features
    
    def _extract_engagement_risk_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract engagement risk indicators"""
        features = {}
        
        try:
            # Time-based engagement
            total_time = (df['created_at'].max() - df['created_at'].min()).total_seconds() / 3600  # hours
            if total_time > 0:
                features['interaction_density'] = len(df) / total_time
            
            # Daily engagement patterns
            daily_interactions = df.groupby(df['created_at'].dt.date).size()
            if len(daily_interactions) > 1:
                features['engagement_consistency'] = 1.0 - (daily_interactions.std() / daily_interactions.mean())
                
                # Detect declining engagement
                recent_days = daily_interactions.tail(7)
                earlier_days = daily_interactions.head(7)
                if len(recent_days) > 0 and len(earlier_days) > 0:
                    features['engagement_trend'] = recent_days.mean() - earlier_days.mean()
            
            # Session abandonment patterns
            if 'execution_time_ms' in df.columns:
                # Detect very short interactions (possible abandonment)
                short_interactions = (df['execution_time_ms'] < 5000).sum()  # Less than 5 seconds
                features['abandonment_rate'] = short_interactions / len(df)
            
            # Weekend vs weekday engagement
            df['is_weekend'] = df['created_at'].dt.dayofweek.isin([5, 6])
            weekend_interactions = df[df['is_weekend']].shape[0]
            weekday_interactions = df[~df['is_weekend']].shape[0]
            
            if weekday_interactions > 0:
                features['weekend_engagement_ratio'] = weekend_interactions / weekday_interactions
            
        except Exception as e:
            logger.error(f"❌ Failed to extract engagement risk features: {e}")
        
        return features
    
    def _extract_temporal_risk_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract temporal pattern risk indicators"""
        features = {}
        
        try:
            # Time-of-day patterns
            df['hour'] = df['created_at'].dt.hour
            
            # Late night studying (potential sign of procrastination or desperation)
            late_night_interactions = ((df['hour'] >= 23) | (df['hour'] <= 6)).sum()
            features['late_night_study_rate'] = late_night_interactions / len(df)
            
            # Cramming behavior (high activity in short time periods)
            if len(df) > 1:
                time_diffs = df['created_at'].diff().dt.total_seconds() / 3600  # hours
                intense_sessions = (time_diffs < 0.1).sum()  # Less than 6 minutes between interactions
                features['cramming_intensity'] = intense_sessions / len(df)
            
            # Study schedule irregularity
            hourly_distribution = df['hour'].value_counts()
            if len(hourly_distribution) > 1:
                # Calculate entropy of hour distribution (high entropy = irregular schedule)
                probabilities = hourly_distribution / hourly_distribution.sum()
                entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
                features['schedule_irregularity'] = entropy / np.log2(24)  # Normalize by max entropy
            
            # Long gaps between sessions
            if len(df) > 1:
                gaps = df['created_at'].diff().dt.total_seconds() / 3600  # hours
                long_gaps = (gaps > 72).sum()  # More than 3 days
                features['long_gap_frequency'] = long_gaps / len(df)
        
        except Exception as e:
            logger.error(f"❌ Failed to extract temporal risk features: {e}")
        
        return features
    
    async def _extract_social_risk_features(self, student_id: str, conn) -> Dict[str, float]:
        """Extract social interaction risk indicators"""
        features = {}
        
        try:
            # Peer comparison metrics
            peer_comparison = await conn.fetchrow("""
                WITH student_performance AS (
                    SELECT AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as student_success_rate
                    FROM interactions 
                    WHERE user_id = $1 AND created_at >= $2
                ),
                class_performance AS (
                    SELECT 
                        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY user_success_rate) as p25,
                        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY user_success_rate) as p50,
                        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY user_success_rate) as p75
                    FROM (
                        SELECT user_id, AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as user_success_rate
                        FROM interactions 
                        WHERE created_at >= $2
                        GROUP BY user_id
                    ) class_stats
                )
                SELECT 
                    s.student_success_rate,
                    c.p25, c.p50, c.p75,
                    CASE 
                        WHEN s.student_success_rate < c.p25 THEN 1.0
                        WHEN s.student_success_rate < c.p50 THEN 0.5
                        ELSE 0.0
                    END as isolation_risk
                FROM student_performance s, class_performance c
            """, student_id, datetime.now() - timedelta(days=30))
            
            if peer_comparison:
                features['peer_isolation_risk'] = peer_comparison['isolation_risk']
                student_rate = peer_comparison['student_success_rate'] or 0.0
                class_median = peer_comparison['p50'] or 0.5
                features['performance_gap'] = max(0.0, class_median - student_rate)
            
            # Help-seeking vs help-giving balance
            help_metrics = await conn.fetchrow("""
                SELECT 
                    COUNT(CASE WHEN metadata::text LIKE '%help_requested%' THEN 1 END) as help_requests,
                    COUNT(CASE WHEN metadata::text LIKE '%help_given%' THEN 1 END) as help_given,
                    COUNT(*) as total_interactions
                FROM interactions 
                WHERE user_id = $1 AND created_at >= $2
            """, student_id, datetime.now() - timedelta(days=30))
            
            if help_metrics and help_metrics['total_interactions'] > 0:
                help_request_rate = help_metrics['help_requests'] / help_metrics['total_interactions']
                help_giving_rate = help_metrics['help_given'] / help_metrics['total_interactions']
                
                features['excessive_help_seeking'] = min(1.0, help_request_rate * 5)  # Scale up
                features['social_contribution'] = help_giving_rate
        
        except Exception as e:
            logger.error(f"❌ Failed to extract social risk features: {e}")
        
        return features
    
    def _extract_technical_risk_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract technical difficulty risk indicators"""
        features = {}
        
        try:
            # Error rate patterns
            technical_errors = 0
            timeout_errors = 0
            
            for _, row in df.iterrows():
                if row.get('metadata'):
                    try:
                        metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
                        error_type = metadata.get('error_type', '')
                        
                        if 'technical' in error_type.lower() or 'system' in error_type.lower():
                            technical_errors += 1
                        elif 'timeout' in error_type.lower():
                            timeout_errors += 1
                    except:
                        pass
            
            if len(df) > 0:
                features['technical_error_rate'] = technical_errors / len(df)
                features['timeout_error_rate'] = timeout_errors / len(df)
            
            # Device/platform consistency
            devices = []
            for _, row in df.iterrows():
                if row.get('metadata'):
                    try:
                        metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
                        device = metadata.get('device_type', 'unknown')
                        devices.append(device)
                    except:
                        devices.append('unknown')
            
            if devices:
                device_changes = sum(1 for i in range(1, len(devices)) if devices[i] != devices[i-1])
                features['device_instability'] = device_changes / len(devices) if len(devices) > 1 else 0.0
        
        except Exception as e:
            logger.error(f"❌ Failed to extract technical risk features: {e}")
        
        return features

    # Risk calculation methods for individual indicators
    async def _calculate_performance_decline_risk(self, student_id: str, 
                                                features: Dict[str, Any], 
                                                horizon: int) -> RiskIndicator:
        """Calculate risk of performance decline"""
        try:
            academic_features = features.get('academic_features', {})
            
            # Base risk from current performance
            current_success = academic_features.get('current_success_rate', 0.5)
            performance_trend = academic_features.get('performance_trend', 0.0)
            performance_volatility = academic_features.get('performance_volatility', 0.0)
            
            # Calculate risk score
            risk_score = 0.0
            
            # Low current performance
            if current_success < 0.6:
                risk_score += (0.6 - current_success) * 2.0
            
            # Negative trend
            if performance_trend < 0:
                risk_score += abs(performance_trend) * 3.0
            
            # High volatility
            if performance_volatility > 0.3:
                risk_score += (performance_volatility - 0.3) * 2.0
            
            risk_score = min(1.0, risk_score)
            
            # Determine trend direction
            if performance_trend > 0.1:
                trend = "improving"
            elif performance_trend < -0.1:
                trend = "declining"
            else:
                trend = "stable"
            
            # Determine severity
            if risk_score >= 0.8:
                severity = RiskLevel.CRITICAL
            elif risk_score >= 0.6:
                severity = RiskLevel.HIGH
            elif risk_score >= 0.4:
                severity = RiskLevel.MEDIUM
            else:
                severity = RiskLevel.LOW
            
            return RiskIndicator(
                indicator_name="performance_decline",
                risk_score=risk_score,
                confidence=0.8,  # High confidence for performance metrics
                trend_direction=trend,
                severity=severity,
                contributing_factors={
                    'current_performance': 1.0 - current_success,
                    'negative_trend': max(0.0, -performance_trend),
                    'volatility': performance_volatility
                },
                time_horizon=horizon,
                reliability_score=0.85
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate performance decline risk: {e}")
            return None
    
    async def _calculate_engagement_drop_risk(self, student_id: str, 
                                            features: Dict[str, Any], 
                                            horizon: int) -> RiskIndicator:
        """Calculate risk of engagement dropping"""
        try:
            engagement_features = features.get('engagement_features', {})
            temporal_features = features.get('temporal_features', {})
            
            # Engagement indicators
            interaction_density = engagement_features.get('interaction_density', 1.0)
            engagement_trend = engagement_features.get('engagement_trend', 0.0)
            abandonment_rate = engagement_features.get('abandonment_rate', 0.0)
            long_gap_frequency = temporal_features.get('long_gap_frequency', 0.0)
            
            # Calculate risk score
            risk_score = 0.0
            
            # Low interaction density
            if interaction_density < 0.5:
                risk_score += (0.5 - interaction_density) * 2.0
            
            # Negative engagement trend
            if engagement_trend < 0:
                risk_score += abs(engagement_trend) * 0.1  # Scale appropriately
            
            # High abandonment rate
            risk_score += abandonment_rate * 2.0
            
            # Frequent long gaps
            risk_score += long_gap_frequency * 3.0
            
            risk_score = min(1.0, risk_score)
            
            # Determine severity
            if risk_score >= 0.75:
                severity = RiskLevel.HIGH
            elif risk_score >= 0.5:
                severity = RiskLevel.MEDIUM
            else:
                severity = RiskLevel.LOW
            
            return RiskIndicator(
                indicator_name="engagement_drop",
                risk_score=risk_score,
                confidence=0.75,
                trend_direction="declining" if engagement_trend < 0 else "stable",
                severity=severity,
                contributing_factors={
                    'low_density': max(0.0, 0.5 - interaction_density),
                    'abandonment': abandonment_rate,
                    'gaps': long_gap_frequency
                },
                time_horizon=horizon,
                reliability_score=0.8
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate engagement drop risk: {e}")
            return None
    
    # Additional risk calculation methods would be implemented here...
    
    async def _run_ensemble_predictions(self, features: Dict[str, Any], 
                                      student_id: str) -> Dict[str, float]:
        """Run ensemble of ML models for risk prediction"""
        try:
            # Flatten features for ML models
            feature_vector = self._flatten_features(features)
            
            if len(feature_vector) == 0:
                return {'ensemble_risk': 0.5}
            
            predictions = {}
            
            # Note: In a real implementation, models would be trained
            # For now, we'll simulate predictions based on feature analysis
            
            # Simple heuristic-based prediction
            behavioral_risk = np.mean([
                features.get('behavioral_features', {}).get('help_seeking_rate', 0.0),
                features.get('behavioral_features', {}).get('rapid_submission_rate', 0.0),
                features.get('behavioral_features', {}).get('impulsive_response_rate', 0.0)
            ])
            
            academic_risk = np.mean([
                1.0 - features.get('academic_features', {}).get('current_success_rate', 0.5),
                max(0.0, -features.get('academic_features', {}).get('performance_trend', 0.0)),
                features.get('academic_features', {}).get('performance_volatility', 0.0)
            ])
            
            engagement_risk = np.mean([
                features.get('engagement_features', {}).get('abandonment_rate', 0.0),
                max(0.0, -features.get('engagement_features', {}).get('engagement_trend', 0.0)),
                features.get('temporal_features', {}).get('long_gap_frequency', 0.0)
            ])
            
            ensemble_risk = np.mean([behavioral_risk, academic_risk, engagement_risk])
            
            predictions = {
                'ensemble_risk': ensemble_risk,
                'behavioral_risk': behavioral_risk,
                'academic_risk': academic_risk,
                'engagement_risk': engagement_risk
            }
            
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Failed to run ensemble predictions: {e}")
            return {'ensemble_risk': 0.5}
    
    def _flatten_features(self, features: Dict[str, Any]) -> List[float]:
        """Flatten nested feature dictionary to list"""
        feature_vector = []
        
        for feature_group in features.values():
            if isinstance(feature_group, dict):
                for value in feature_group.values():
                    if isinstance(value, (int, float)) and not np.isnan(float(value)):
                        feature_vector.append(float(value))
        
        return feature_vector
    
    async def _calculate_overall_risk(self, risk_indicators: List[RiskIndicator],
                                    ensemble_predictions: Dict[str, float]) -> float:
        """Calculate overall risk score from individual indicators and ensemble"""
        try:
            if not risk_indicators and not ensemble_predictions:
                return 0.5  # Default medium risk
            
            # Weight individual risk indicators
            indicator_scores = []
            for indicator in risk_indicators:
                weighted_score = indicator.risk_score * indicator.reliability_score
                indicator_scores.append(weighted_score)
            
            # Combine with ensemble predictions
            ensemble_risk = ensemble_predictions.get('ensemble_risk', 0.5)
            
            # Calculate weighted average
            if indicator_scores:
                indicator_avg = np.mean(indicator_scores)
                # Weight: 60% indicators, 40% ensemble
                overall_risk = 0.6 * indicator_avg + 0.4 * ensemble_risk
            else:
                overall_risk = ensemble_risk
            
            return min(1.0, max(0.0, overall_risk))
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate overall risk: {e}")
            return 0.5
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from risk score"""
        if risk_score >= self.config['risk_thresholds']['critical']:
            return RiskLevel.CRITICAL
        elif risk_score >= self.config['risk_thresholds']['high']:
            return RiskLevel.HIGH
        elif risk_score >= self.config['risk_thresholds']['medium']:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _calculate_confidence_bounds(self, ensemble_predictions: Dict[str, float],
                                   risk_indicators: List[RiskIndicator]) -> Tuple[float, float]:
        """Calculate confidence bounds for risk prediction"""
        try:
            # Simple confidence bound calculation
            # In practice, this would use more sophisticated uncertainty quantification
            
            base_confidence = 0.1  # Base uncertainty
            
            # Reduce confidence if we have fewer indicators
            if len(risk_indicators) < 3:
                base_confidence += 0.1
            
            # Reduce confidence if indicators disagree significantly
            if len(risk_indicators) > 1:
                scores = [ind.risk_score for ind in risk_indicators]
                score_std = np.std(scores)
                if score_std > 0.3:
                    base_confidence += 0.15
            
            ensemble_risk = ensemble_predictions.get('ensemble_risk', 0.5)
            
            lower_bound = max(0.0, ensemble_risk - base_confidence)
            upper_bound = min(1.0, ensemble_risk + base_confidence)
            
            return (lower_bound, upper_bound)
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate confidence bounds: {e}")
            return (0.0, 1.0)
    
    async def _predict_specific_outcomes(self, student_id: str, features: Dict[str, Any],
                                       horizon: int) -> Dict[str, float]:
        """Predict specific educational outcomes"""
        try:
            outcomes = {}
            
            # Performance predictions
            current_success = features.get('academic_features', {}).get('current_success_rate', 0.5)
            performance_trend = features.get('academic_features', {}).get('performance_trend', 0.0)
            
            # Project performance trend
            future_performance = current_success + (performance_trend * horizon / 7.0)
            outcomes['predicted_success_rate'] = max(0.0, min(1.0, future_performance))
            
            # Engagement predictions
            engagement_trend = features.get('engagement_features', {}).get('engagement_trend', 0.0)
            current_density = features.get('engagement_features', {}).get('interaction_density', 1.0)
            
            future_engagement = current_density + (engagement_trend * horizon / 7.0)
            outcomes['predicted_engagement_level'] = max(0.0, min(2.0, future_engagement))
            
            # Dropout risk
            abandonment_rate = features.get('engagement_features', {}).get('abandonment_rate', 0.0)
            long_gaps = features.get('temporal_features', {}).get('long_gap_frequency', 0.0)
            
            dropout_risk = (abandonment_rate + long_gaps + max(0.0, -performance_trend)) / 3.0
            outcomes['dropout_risk'] = min(1.0, dropout_risk)
            
            # Time to mastery (for current concepts)
            if current_success > 0:
                # Simple estimation based on current performance
                mastery_threshold = 0.8
                if current_success >= mastery_threshold:
                    outcomes['days_to_mastery'] = 0
                else:
                    improvement_needed = mastery_threshold - current_success
                    if performance_trend > 0:
                        outcomes['days_to_mastery'] = improvement_needed / (performance_trend + 0.01) * 7
                    else:
                        outcomes['days_to_mastery'] = 999  # Unlikely to achieve mastery
            
            return outcomes
            
        except Exception as e:
            logger.error(f"❌ Failed to predict specific outcomes: {e}")
            return {}
    
    def _create_default_risk_profile(self, student_id: str) -> StudentRiskProfile:
        """Create default risk profile when assessment fails"""
        return StudentRiskProfile(
            student_id=student_id,
            overall_risk_score=0.5,
            risk_level=RiskLevel.MEDIUM,
            risk_indicators=[],
            predicted_outcomes={},
            intervention_recommendations=[],
            confidence_bounds=(0.3, 0.7)
        )
    
    async def _store_risk_assessment(self, risk_profile: StudentRiskProfile):
        """Store risk assessment in database"""
        try:
            if not self.db_manager:
                return
            
            async with self.db_manager.postgres.get_connection() as conn:
                # Store overall assessment
                await conn.execute("""
                    INSERT INTO risk_assessments 
                    (student_id, overall_risk_score, risk_level, assessment_date, predicted_outcomes)
                    VALUES ($1, $2, $3, $4, $5)
                """, risk_profile.student_id, risk_profile.overall_risk_score,
                    risk_profile.risk_level.value, risk_profile.assessment_timestamp,
                    json.dumps(risk_profile.predicted_outcomes))
                
                # Store individual risk indicators
                for indicator in risk_profile.risk_indicators:
                    await conn.execute("""
                        INSERT INTO risk_indicators 
                        (student_id, indicator_name, risk_score, severity, assessment_date)
                        VALUES ($1, $2, $3, $4, $5)
                    """, risk_profile.student_id, indicator.indicator_name,
                        indicator.risk_score, indicator.severity.value,
                        risk_profile.assessment_timestamp)
        
        except Exception as e:
            logger.error(f"❌ Failed to store risk assessment: {e}")
    
    async def _check_and_generate_alerts(self, risk_profile: StudentRiskProfile):
        """Check if alerts should be generated and create them"""
        try:
            alerts_to_create = []
            
            # Critical risk level always generates alert
            if risk_profile.risk_level == RiskLevel.CRITICAL:
                alert = await self._create_alert(risk_profile, "critical_risk_detected")
                alerts_to_create.append(alert)
            
            # High risk in specific areas
            for indicator in risk_profile.risk_indicators:
                if indicator.severity in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                    alert = await self._create_alert(risk_profile, f"high_{indicator.indicator_name}")
                    alerts_to_create.append(alert)
            
            # Store and process alerts
            for alert in alerts_to_create:
                await self.alert_manager.create_alert(alert)
                
        except Exception as e:
            logger.error(f"❌ Failed to check and generate alerts: {e}")
    
    async def _create_alert(self, risk_profile: StudentRiskProfile, 
                          alert_type: str) -> EarlyWarningAlert:
        """Create an early warning alert"""
        try:
            alert_id = f"alert_{risk_profile.student_id}_{alert_type}_{datetime.now().timestamp()}"
            
            # Determine alert triggers
            triggers = {}
            for indicator in risk_profile.risk_indicators:
                if indicator.severity in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                    triggers[indicator.indicator_name] = indicator.risk_score
            
            # Get top intervention recommendations
            top_interventions = sorted(
                risk_profile.intervention_recommendations,
                key=lambda x: x.priority_level,
                reverse=True
            )[:3]
            
            alert = EarlyWarningAlert(
                alert_id=alert_id,
                student_id=risk_profile.student_id,
                alert_type=alert_type,
                severity=risk_profile.risk_level,
                risk_indicators=[ind.indicator_name for ind in risk_profile.risk_indicators],
                predicted_outcome=f"Risk level: {risk_profile.risk_level.value} ({risk_profile.overall_risk_score:.2f})",
                confidence=risk_profile.confidence_bounds[1] - risk_profile.confidence_bounds[0],
                intervention_recommendations=top_interventions,
                alert_triggers=triggers
            )
            
            return alert
            
        except Exception as e:
            logger.error(f"❌ Failed to create alert: {e}")
            return None

class InterventionEngine:
    """Engine for generating personalized intervention recommendations"""
    
    def __init__(self):
        self.intervention_strategies = {}
        self.intervention_effectiveness = defaultdict(list)
    
    async def initialize(self):
        """Initialize intervention strategies"""
        try:
            self.intervention_strategies = {
                'performance_decline': [
                    {
                        'type': InterventionType.ACADEMIC_SUPPORT,
                        'description': 'Provide additional practice problems with guided solutions',
                        'priority': 4,
                        'expected_impact': 0.3,
                        'effort': 'medium'
                    },
                    {
                        'type': InterventionType.DIFFICULTY_ADJUSTMENT,
                        'description': 'Temporarily reduce problem difficulty to build confidence',
                        'priority': 3,
                        'expected_impact': 0.2,
                        'effort': 'low'
                    }
                ],
                'engagement_drop': [
                    {
                        'type': InterventionType.MOTIVATIONAL,
                        'description': 'Send encouraging messages and highlight progress',
                        'priority': 3,
                        'expected_impact': 0.25,
                        'effort': 'low'
                    },
                    {
                        'type': InterventionType.LEARNING_STRATEGY,
                        'description': 'Suggest gamified learning activities',
                        'priority': 2,
                        'expected_impact': 0.2,
                        'effort': 'medium'
                    }
                ],
                'help_seeking_excess': [
                    {
                        'type': InterventionType.LEARNING_STRATEGY,
                        'description': 'Teach self-debugging and problem-solving strategies',
                        'priority': 4,
                        'expected_impact': 0.4,
                        'effort': 'high'
                    }
                ]
            }
            
            logger.info("✅ Intervention engine initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize intervention engine: {e}")
    
    async def generate_recommendations(self, student_id: str, 
                                     risk_indicators: List[RiskIndicator],
                                     overall_risk: float) -> List[InterventionRecommendation]:
        """Generate personalized intervention recommendations"""
        try:
            recommendations = []
            
            for indicator in risk_indicators:
                if indicator.severity in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                    strategies = self.intervention_strategies.get(indicator.indicator_name, [])
                    
                    for strategy in strategies:
                        intervention_id = f"intervention_{student_id}_{indicator.indicator_name}_{len(recommendations)}"
                        
                        recommendation = InterventionRecommendation(
                            intervention_id=intervention_id,
                            intervention_type=strategy['type'],
                            priority_level=strategy['priority'],
                            description=strategy['description'],
                            rationale=f"Addressing {indicator.indicator_name} (risk: {indicator.risk_score:.2f})",
                            expected_impact=strategy['expected_impact'],
                            urgency_score=indicator.risk_score,
                            effort_required=strategy['effort'],
                            success_probability=0.7,  # Default
                            implementation_steps=[
                                "Assess current student state",
                                "Implement intervention",
                                "Monitor progress"
                            ],
                            monitoring_metrics=[
                                indicator.indicator_name,
                                "overall_performance"
                            ]
                        )
                        
                        recommendations.append(recommendation)
            
            # Sort by priority and urgency
            recommendations.sort(key=lambda x: (x.priority_level, x.urgency_score), reverse=True)
            
            return recommendations[:5]  # Return top 5 recommendations
            
        except Exception as e:
            logger.error(f"❌ Failed to generate intervention recommendations: {e}")
            return []

class AlertManager:
    """Manager for handling early warning alerts"""
    
    def __init__(self):
        self.active_alerts = {}
        self.alert_history = defaultdict(list)
    
    async def create_alert(self, alert: EarlyWarningAlert):
        """Create and store a new alert"""
        try:
            self.active_alerts[alert.alert_id] = alert
            self.alert_history[alert.student_id].append(alert)
            
            logger.info(f"🚨 Created {alert.severity.value} alert for student {alert.student_id}: {alert.alert_type}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create alert: {e}")
    
    async def get_active_alerts(self, student_id: Optional[str] = None) -> List[EarlyWarningAlert]:
        """Get active alerts for a student or all students"""
        try:
            if student_id:
                return [alert for alert in self.active_alerts.values() 
                       if alert.student_id == student_id and not alert.resolved]
            else:
                return [alert for alert in self.active_alerts.values() if not alert.resolved]
                
        except Exception as e:
            logger.error(f"❌ Failed to get active alerts: {e}")
            return []

# Testing function
async def test_advanced_early_warning_system():
    """Test the advanced early warning system"""
    try:
        logger.info("🧪 Testing Advanced Early Warning System")
        
        system = AdvancedEarlyWarningSystem()
        await system.initialize()
        
        # Test risk assessment
        risk_profile = await system.assess_student_risk("test_student")
        logger.info(f"✅ Risk assessment completed: {risk_profile.risk_level.value}")
        
        # Test alert generation
        alerts = await system.alert_manager.get_active_alerts("test_student")
        logger.info(f"✅ Generated {len(alerts)} alerts")
        
        logger.info("✅ Advanced Early Warning System test completed")
        
    except Exception as e:
        logger.error(f"❌ Advanced Early Warning System test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_advanced_early_warning_system())