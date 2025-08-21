#!/usr/bin/env python3
"""
Advanced ML Engine for Physics Assistant Phase 6
Comprehensive machine learning system with adaptive learning, predictive analytics,
and personalized recommendations using neural networks and advanced algorithms.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import xgboost as xgb
import lightgbm as lgb
from transformers import AutoTokenizer, AutoModel
import cv2
import pytesseract
from PIL import Image
import networkx as nx
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import pickle
import joblib
import warnings
import os
from concurrent.futures import ThreadPoolExecutor
import redis
import hashlib

# Suppress warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MLModelConfig:
    """ML Model configuration"""
    model_type: str
    hyperparameters: Dict[str, Any]
    training_config: Dict[str, Any]
    deployment_config: Dict[str, Any]
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class StudentProfile:
    """Enhanced student profile with ML features"""
    student_id: str
    learning_style_vector: np.ndarray
    knowledge_state_vector: np.ndarray
    behavioral_patterns: Dict[str, float]
    engagement_metrics: Dict[str, float]
    misconception_patterns: List[str]
    learning_velocity: float
    preferred_modalities: List[str]
    cognitive_load_capacity: float
    metacognitive_skills: float
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class PredictionOutput:
    """ML prediction output with uncertainty quantification"""
    prediction: float
    confidence: float
    uncertainty_bounds: Tuple[float, float]
    feature_importance: Dict[str, float]
    explanation: str
    model_version: str
    prediction_timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class LearningRecommendation:
    """Personalized learning recommendation"""
    student_id: str
    recommended_action: str
    reasoning: str
    confidence: float
    expected_outcome: str
    alternative_actions: List[str]
    personalization_factors: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)

class AdaptiveLearningNet(nn.Module):
    """PyTorch neural network for adaptive learning prediction"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout_rate: float = 0.3):
        super(AdaptiveLearningNet, self).__init__()
        
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
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Uncertainty estimation layers
        self.uncertainty_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, output_dim),
            nn.Softplus()  # Ensure positive uncertainty values
        )
    
    def forward(self, x):
        features = self.network[:-1](x)  # All layers except the last
        predictions = self.network[-1](features)
        uncertainties = self.uncertainty_head(features)
        return predictions, uncertainties

class KnowledgeStateTracker(nn.Module):
    """LSTM-based knowledge state tracking"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_concepts: int, num_layers: int = 2):
        super(KnowledgeStateTracker, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_concepts = num_concepts
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        
        # Knowledge state prediction heads
        self.mastery_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_concepts),
            nn.Sigmoid()  # Mastery probabilities [0, 1]
        )
        
        self.forgetting_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_concepts),
            nn.Sigmoid()  # Forgetting rates [0, 1]
        )
    
    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Take the last output for predictions
        last_output = lstm_out[:, -1, :]
        
        mastery_probs = self.mastery_head(last_output)
        forgetting_rates = self.forgetting_head(last_output)
        
        return mastery_probs, forgetting_rates, hidden

class AdvancedMLEngine:
    """Advanced ML engine with deep learning and adaptive capabilities"""
    
    def __init__(self, db_manager=None, redis_client=None, model_storage_path="/tmp/ml_models"):
        self.db_manager = db_manager
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, db=0)
        self.model_storage_path = model_storage_path
        
        # ML Models registry
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.model_configs = {}
        
        # Neural networks
        self.adaptive_learning_net = None
        self.knowledge_tracker = None
        
        # Feature extractors
        self.text_tokenizer = None
        self.text_encoder = None
        
        # Configuration
        self.config = {
            'adaptive_learning': {
                'hidden_dims': [256, 128, 64],
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100
            },
            'knowledge_tracking': {
                'hidden_dim': 128,
                'num_layers': 2,
                'learning_rate': 0.001,
                'sequence_length': 50
            },
            'engagement_prediction': {
                'model_type': 'xgboost',
                'n_estimators': 200,
                'max_depth': 8,
                'learning_rate': 0.1
            },
            'recommendation_system': {
                'embedding_dim': 64,
                'num_factors': 32,
                'reg_lambda': 0.01
            }
        }
        
        # Caching configuration
        self.cache_ttl = 3600  # 1 hour
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def initialize(self):
        """Initialize the advanced ML engine"""
        try:
            logger.info("🚀 Initializing Advanced ML Engine")
            
            # Create model storage directory
            os.makedirs(self.model_storage_path, exist_ok=True)
            
            # Initialize text processing models
            await self._initialize_text_models()
            
            # Load existing models
            await self._load_existing_models()
            
            # Initialize neural networks
            await self._initialize_neural_networks()
            
            # Initialize feature extractors
            await self._initialize_feature_extractors()
            
            # Verify GPU availability
            self._check_gpu_availability()
            
            logger.info("✅ Advanced ML Engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Advanced ML Engine: {e}")
            return False
    
    async def _initialize_text_models(self):
        """Initialize text processing models"""
        try:
            # Load pre-trained transformer for text understanding
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.text_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.text_encoder = AutoModel.from_pretrained(model_name)
            
            logger.info("✅ Text processing models initialized")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize text models: {e}")
            self.text_tokenizer = None
            self.text_encoder = None
    
    async def _load_existing_models(self):
        """Load previously trained models"""
        try:
            import glob
            
            model_files = glob.glob(f"{self.model_storage_path}/*.pkl")
            
            for model_file in model_files:
                model_name = os.path.basename(model_file).replace('.pkl', '')
                try:
                    with open(model_file, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    self.models[model_name] = model_data.get('model')
                    self.scalers[model_name] = model_data.get('scaler')
                    self.encoders[model_name] = model_data.get('encoders')
                    self.model_configs[model_name] = model_data.get('config')
                    
                    logger.info(f"📊 Loaded model: {model_name}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load model {model_name}: {e}")
            
            logger.info(f"✅ Loaded {len(model_files)} existing models")
            
        except Exception as e:
            logger.error(f"❌ Failed to load existing models: {e}")
    
    async def _initialize_neural_networks(self):
        """Initialize PyTorch neural networks"""
        try:
            # Get feature dimensions from data analysis
            input_dim = await self._estimate_feature_dimensions()
            
            # Initialize adaptive learning network
            self.adaptive_learning_net = AdaptiveLearningNet(
                input_dim=input_dim,
                hidden_dims=self.config['adaptive_learning']['hidden_dims'],
                output_dim=1,  # Single output for success prediction
                dropout_rate=self.config['adaptive_learning']['dropout_rate']
            )
            
            # Initialize knowledge state tracker
            num_concepts = await self._get_num_concepts()
            self.knowledge_tracker = KnowledgeStateTracker(
                input_dim=input_dim,
                hidden_dim=self.config['knowledge_tracking']['hidden_dim'],
                num_concepts=num_concepts,
                num_layers=self.config['knowledge_tracking']['num_layers']
            )
            
            # Check for GPU and move models
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.adaptive_learning_net.to(device)
            self.knowledge_tracker.to(device)
            
            logger.info(f"✅ Neural networks initialized on {device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize neural networks: {e}")
    
    async def _initialize_feature_extractors(self):
        """Initialize feature extraction components"""
        try:
            # Initialize scalers for different feature groups
            feature_groups = ['behavioral', 'temporal', 'performance', 'content', 'contextual']
            
            for group in feature_groups:
                if group not in self.scalers:
                    self.scalers[group] = StandardScaler()
            
            logger.info("✅ Feature extractors initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize feature extractors: {e}")
    
    def _check_gpu_availability(self):
        """Check GPU availability for acceleration"""
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"🚀 GPU acceleration available: {gpu_count} x {gpu_name}")
        else:
            logger.info("💻 Using CPU for ML computations")
        
        # Check TensorFlow GPU
        if tf.config.list_physical_devices('GPU'):
            logger.info("🚀 TensorFlow GPU support detected")
    
    async def _estimate_feature_dimensions(self) -> int:
        """Estimate feature dimensions from sample data"""
        try:
            if self.db_manager:
                # Get sample interaction data to estimate feature dimensions
                async with self.db_manager.postgres.get_connection() as conn:
                    sample = await conn.fetchrow("""
                        SELECT user_id FROM interactions 
                        WHERE created_at >= NOW() - INTERVAL '7 days'
                        LIMIT 1
                    """)
                    
                    if sample:
                        sample_features = await self.extract_comprehensive_features(str(sample['user_id']))
                        return len(sample_features.get('feature_vector', []))
            
            # Default feature dimension
            return 50
            
        except Exception as e:
            logger.error(f"❌ Failed to estimate feature dimensions: {e}")
            return 50
    
    async def _get_num_concepts(self) -> int:
        """Get number of concepts in the knowledge graph"""
        try:
            if self.db_manager:
                concepts = await self.db_manager.neo4j.run_query("MATCH (c:Concept) RETURN count(c) as count")
                return concepts[0]['count'] if concepts else 10
            
            return 10  # Default number of concepts
            
        except Exception as e:
            logger.error(f"❌ Failed to get number of concepts: {e}")
            return 10
    
    async def extract_comprehensive_features(self, student_id: str, lookback_days: int = 30) -> Dict[str, Any]:
        """Extract comprehensive feature set for ML models"""
        try:
            features = {
                'behavioral_features': {},
                'temporal_features': {},
                'performance_features': {},
                'content_features': {},
                'social_features': {},
                'contextual_features': {},
                'feature_vector': []
            }
            
            if not self.db_manager:
                return features
            
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
                
                df = pd.DataFrame([dict(row) for row in interactions])
                df['created_at'] = pd.to_datetime(df['created_at'])
                
                # Extract different feature groups
                features['behavioral_features'] = await self._extract_behavioral_features(df, student_id)
                features['temporal_features'] = await self._extract_temporal_features(df)
                features['performance_features'] = await self._extract_performance_features(df, student_id)
                features['content_features'] = await self._extract_content_features(df)
                features['social_features'] = await self._extract_social_features(student_id, conn)
                features['contextual_features'] = await self._extract_contextual_features(df, student_id)
                
                # Create unified feature vector
                features['feature_vector'] = self._create_feature_vector(features)
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Failed to extract comprehensive features for {student_id}: {e}")
            return {'behavioral_features': {}, 'temporal_features': {}, 'performance_features': {}, 
                   'content_features': {}, 'social_features': {}, 'contextual_features': {}, 'feature_vector': []}
    
    async def _extract_behavioral_features(self, df: pd.DataFrame, student_id: str) -> Dict[str, float]:
        """Extract behavioral pattern features"""
        features = {}
        
        try:
            total_interactions = len(df)
            if total_interactions == 0:
                return features
            
            # Session patterns
            df['time_diff'] = df['created_at'].diff().dt.total_seconds() / 60
            session_breaks = df['time_diff'] > 30
            
            features['avg_session_duration'] = df.groupby(session_breaks.cumsum())['time_diff'].sum().mean()
            features['session_frequency'] = session_breaks.sum() + 1
            features['interactions_per_session'] = total_interactions / features['session_frequency']
            
            # Response patterns
            if 'execution_time_ms' in df.columns:
                features['avg_response_time'] = df['execution_time_ms'].mean()
                features['response_time_variability'] = df['execution_time_ms'].std()
                features['response_time_trend'] = self._calculate_trend(df['execution_time_ms'].values)
            
            # Help-seeking behavior
            help_requests = 0
            hint_usage = 0
            
            for _, row in df.iterrows():
                if row.get('metadata'):
                    try:
                        metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
                        if metadata.get('help_requested'):
                            help_requests += 1
                        if metadata.get('hint_used'):
                            hint_usage += 1
                    except:
                        pass
            
            features['help_seeking_rate'] = help_requests / total_interactions
            features['hint_usage_rate'] = hint_usage / total_interactions
            
            # Engagement patterns
            features['interaction_intensity'] = total_interactions / ((df['created_at'].max() - df['created_at'].min()).days + 1)
            
            # Error patterns
            if 'success' in df.columns:
                features['error_rate'] = 1.0 - df['success'].mean()
                features['consecutive_errors'] = self._calculate_max_consecutive(df['success'].values, 0)
                features['error_recovery_rate'] = self._calculate_error_recovery_rate(df['success'].values)
            
            # Persistence patterns
            features['session_completion_rate'] = self._calculate_session_completion_rate(df)
            features['retry_behavior'] = self._calculate_retry_behavior(df)
            
        except Exception as e:
            logger.error(f"❌ Failed to extract behavioral features: {e}")
        
        return features
    
    async def _extract_temporal_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract temporal pattern features"""
        features = {}
        
        try:
            # Basic temporal patterns
            df['hour'] = df['created_at'].dt.hour
            df['day_of_week'] = df['created_at'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            features['preferred_hour'] = df['hour'].mode().iloc[0] if not df.empty else 12
            features['weekend_preference'] = df['is_weekend'].mean()
            features['temporal_consistency'] = 1.0 - (df['hour'].std() / 24.0)
            
            # Study pattern analysis
            daily_interactions = df.groupby(df['created_at'].dt.date).size()
            features['study_frequency'] = len(daily_interactions)
            features['study_consistency'] = 1.0 - (daily_interactions.std() / daily_interactions.mean()) if daily_interactions.std() > 0 else 1.0
            
            # Peak performance times
            hourly_success = df.groupby('hour')['success'].mean() if 'success' in df.columns else pd.Series()
            if len(hourly_success) > 0:
                features['peak_performance_hour'] = hourly_success.idxmax()
                features['performance_hour_variance'] = hourly_success.std()
            
            # Time since last interaction
            if len(df) > 0:
                features['hours_since_last_interaction'] = (datetime.now() - df['created_at'].iloc[-1]).total_seconds() / 3600
            
        except Exception as e:
            logger.error(f"❌ Failed to extract temporal features: {e}")
        
        return features
    
    async def _extract_performance_features(self, df: pd.DataFrame, student_id: str) -> Dict[str, float]:
        """Extract academic performance features"""
        features = {}
        
        try:
            if 'success' in df.columns and len(df) > 0:
                # Basic performance
                features['overall_success_rate'] = df['success'].mean()
                features['recent_success_rate'] = df['success'].tail(10).mean() if len(df) >= 10 else df['success'].mean()
                
                # Performance trends
                features['performance_trend'] = self._calculate_trend(df['success'].astype(int).values)
                features['performance_volatility'] = df['success'].rolling(window=5).std().mean()
                
                # Learning velocity
                if len(df) >= 20:
                    early_performance = df['success'].head(10).mean()
                    recent_performance = df['success'].tail(10).mean()
                    features['learning_velocity'] = recent_performance - early_performance
                
                # Difficulty adaptation
                if 'agent_type' in df.columns:
                    difficulty_map = {'math': 1, 'kinematics': 2, 'forces': 3, 'energy': 4, 'momentum': 4, 'angular_motion': 5}
                    df['difficulty'] = df['agent_type'].map(difficulty_map).fillna(2)
                    
                    features['avg_difficulty_attempted'] = df['difficulty'].mean()
                    features['difficulty_progression'] = self._calculate_trend(df['difficulty'].values)
                    
                    # Success by difficulty
                    difficulty_success = df.groupby('difficulty')['success'].mean()
                    for diff, success in difficulty_success.items():
                        features[f'success_rate_difficulty_{int(diff)}'] = success
                
                # Mastery indicators
                features['mastery_consistency'] = self._calculate_mastery_consistency(df['success'].values)
                features['plateau_detection'] = self._detect_learning_plateau(df['success'].values)
            
        except Exception as e:
            logger.error(f"❌ Failed to extract performance features: {e}")
        
        return features
    
    async def _extract_content_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract content interaction features"""
        features = {}
        
        try:
            if 'agent_type' in df.columns:
                # Content diversity
                unique_concepts = df['agent_type'].nunique()
                features['concept_diversity'] = unique_concepts
                
                concept_distribution = df['agent_type'].value_counts(normalize=True)
                features['content_concentration'] = np.sum(concept_distribution ** 2)  # Herfindahl index
                
                # Concept switching behavior
                concept_switches = (df['agent_type'] != df['agent_type'].shift()).sum() - 1
                features['concept_switching_rate'] = concept_switches / len(df) if len(df) > 1 else 0
                
                # Topic mastery progression
                if 'success' in df.columns:
                    for concept in df['agent_type'].unique():
                        concept_data = df[df['agent_type'] == concept]
                        if len(concept_data) >= 3:
                            concept_trend = self._calculate_trend(concept_data['success'].astype(int).values)
                            features[f'{concept}_mastery_trend'] = concept_trend
            
            # Problem complexity analysis
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
                features['complexity_progression'] = self._calculate_trend(np.array(complexity_scores))
            
        except Exception as e:
            logger.error(f"❌ Failed to extract content features: {e}")
        
        return features
    
    async def _extract_social_features(self, student_id: str, conn) -> Dict[str, float]:
        """Extract social learning features"""
        features = {}
        
        try:
            # Peer comparison metrics
            peer_stats = await conn.fetchrow("""
                WITH student_stats AS (
                    SELECT AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as student_success_rate
                    FROM interactions 
                    WHERE user_id = $1 AND created_at >= NOW() - INTERVAL '30 days'
                ),
                peer_stats AS (
                    SELECT 
                        AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as peer_avg_success,
                        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY CASE WHEN success THEN 1.0 ELSE 0.0 END) as peer_median_success
                    FROM interactions 
                    WHERE user_id != $1 AND created_at >= NOW() - INTERVAL '30 days'
                )
                SELECT 
                    s.student_success_rate,
                    p.peer_avg_success,
                    p.peer_median_success,
                    s.student_success_rate - p.peer_avg_success as performance_vs_peers
                FROM student_stats s, peer_stats p
            """, student_id)
            
            if peer_stats:
                features['performance_vs_peers'] = peer_stats['performance_vs_peers'] or 0.0
                features['peer_percentile'] = self._calculate_percentile_rank(
                    peer_stats['student_success_rate'] or 0.0,
                    peer_stats['peer_avg_success'] or 0.0,
                    peer_stats['peer_median_success'] or 0.0
                )
            
            # Collaborative learning indicators
            help_given = await conn.fetchval("""
                SELECT COUNT(*) FROM interactions 
                WHERE user_id = $1 
                AND metadata::text LIKE '%help_given%'
                AND created_at >= NOW() - INTERVAL '30 days'
            """, student_id)
            
            features['help_giving_behavior'] = help_given or 0
            
        except Exception as e:
            logger.error(f"❌ Failed to extract social features: {e}")
        
        return features
    
    async def _extract_contextual_features(self, df: pd.DataFrame, student_id: str) -> Dict[str, float]:
        """Extract contextual learning features"""
        features = {}
        
        try:
            # Device and environment context
            device_types = []
            for _, row in df.iterrows():
                if row.get('metadata'):
                    try:
                        metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
                        device = metadata.get('device_type', 'desktop')
                        device_types.append(device)
                    except:
                        device_types.append('desktop')
            
            if device_types:
                device_distribution = pd.Series(device_types).value_counts(normalize=True)
                features['primary_device'] = device_distribution.index[0] if len(device_distribution) > 0 else 'desktop'
                features['device_switching'] = len(device_distribution)
            
            # Study environment consistency
            if 'success' in df.columns:
                success_by_hour = df.groupby(df['created_at'].dt.hour)['success'].mean()
                features['environment_consistency'] = 1.0 - success_by_hour.std() if len(success_by_hour) > 1 else 1.0
            
        except Exception as e:
            logger.error(f"❌ Failed to extract contextual features: {e}")
        
        return features
    
    def _create_feature_vector(self, features: Dict[str, Dict[str, float]]) -> List[float]:
        """Create unified feature vector from all feature groups"""
        try:
            feature_vector = []
            
            # Combine all numerical features
            for feature_group in ['behavioral_features', 'temporal_features', 'performance_features', 
                                'content_features', 'social_features', 'contextual_features']:
                group_features = features.get(feature_group, {})
                for key, value in group_features.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        feature_vector.append(float(value))
            
            # Pad or truncate to fixed size
            target_size = 50
            if len(feature_vector) > target_size:
                feature_vector = feature_vector[:target_size]
            elif len(feature_vector) < target_size:
                feature_vector.extend([0.0] * (target_size - len(feature_vector)))
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"❌ Failed to create feature vector: {e}")
            return [0.0] * 50
    
    # Helper methods for feature extraction
    def _calculate_trend(self, values: np.ndarray) -> float:
        """Calculate trend using linear regression slope"""
        if len(values) < 2:
            return 0.0
        
        try:
            x = np.arange(len(values))
            z = np.polyfit(x, values, 1)
            return float(z[0])
        except:
            return 0.0
    
    def _calculate_max_consecutive(self, values: np.ndarray, target_value: int) -> int:
        """Calculate maximum consecutive occurrences of target value"""
        if len(values) == 0:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for value in values:
            if value == target_value:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_error_recovery_rate(self, success_values: np.ndarray) -> float:
        """Calculate rate of recovery from errors"""
        if len(success_values) < 2:
            return 0.0
        
        recoveries = 0
        opportunities = 0
        
        for i in range(len(success_values) - 1):
            if success_values[i] == 0:  # Error occurred
                opportunities += 1
                if success_values[i + 1] == 1:  # Recovered in next attempt
                    recoveries += 1
        
        return recoveries / opportunities if opportunities > 0 else 0.0
    
    def _calculate_session_completion_rate(self, df: pd.DataFrame) -> float:
        """Calculate rate of session completion"""
        # Simplified implementation
        return 0.8  # Placeholder
    
    def _calculate_retry_behavior(self, df: pd.DataFrame) -> float:
        """Calculate retry behavior patterns"""
        # Simplified implementation
        return 0.3  # Placeholder
    
    def _calculate_mastery_consistency(self, success_values: np.ndarray) -> float:
        """Calculate consistency of mastery demonstration"""
        if len(success_values) < 5:
            return 0.0
        
        # Use rolling window to check consistency
        window_size = min(5, len(success_values))
        rolling_means = []
        
        for i in range(len(success_values) - window_size + 1):
            window = success_values[i:i + window_size]
            rolling_means.append(np.mean(window))
        
        return 1.0 - np.std(rolling_means) if len(rolling_means) > 0 else 0.0
    
    def _detect_learning_plateau(self, success_values: np.ndarray) -> float:
        """Detect if student has reached a learning plateau"""
        if len(success_values) < 10:
            return 0.0
        
        # Check last 10 values for plateau (low variance)
        recent_values = success_values[-10:]
        variance = np.var(recent_values)
        
        # Lower variance indicates plateau
        return 1.0 - variance if variance < 1.0 else 0.0
    
    def _calculate_percentile_rank(self, student_score: float, peer_avg: float, peer_median: float) -> float:
        """Calculate percentile rank relative to peers"""
        if student_score >= peer_avg:
            # Above average, interpolate between 50th and 100th percentile
            return 50.0 + 50.0 * min(1.0, (student_score - peer_avg) / (1.0 - peer_avg))
        else:
            # Below average, interpolate between 0th and 50th percentile
            return 50.0 * (student_score / peer_avg) if peer_avg > 0 else 0.0
    
    async def save_model(self, model_name: str, model, scaler=None, encoders=None, config=None):
        """Save trained model to storage"""
        try:
            model_data = {
                'model': model,
                'scaler': scaler,
                'encoders': encoders,
                'config': config,
                'timestamp': datetime.now(),
                'version': '1.0'
            }
            
            model_path = f"{self.model_storage_path}/{model_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Store model metadata in database if available
            if self.db_manager:
                await self._store_model_metadata(model_name, model_data)
            
            logger.info(f"💾 Saved model: {model_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save model {model_name}: {e}")
    
    async def _store_model_metadata(self, model_name: str, model_data: Dict[str, Any]):
        """Store model metadata in database"""
        try:
            async with self.db_manager.postgres.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO ml_models (name, version, config, created_at)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (name) DO UPDATE SET
                        version = EXCLUDED.version,
                        config = EXCLUDED.config,
                        updated_at = EXCLUDED.created_at
                """, model_name, model_data['version'], 
                    json.dumps(model_data.get('config', {})), model_data['timestamp'])
                
        except Exception as e:
            logger.error(f"❌ Failed to store model metadata: {e}")

# Main testing function
async def test_advanced_ml_engine():
    """Test advanced ML engine"""
    try:
        logger.info("🧪 Testing Advanced ML Engine")
        
        engine = AdvancedMLEngine()
        await engine.initialize()
        
        # Test feature extraction
        sample_features = await engine.extract_comprehensive_features("test_student")
        logger.info(f"✅ Extracted {len(sample_features['feature_vector'])} features")
        
        logger.info("✅ Advanced ML Engine test completed")
        
    except Exception as e:
        logger.error(f"❌ Advanced ML Engine test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_advanced_ml_engine())