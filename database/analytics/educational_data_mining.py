#!/usr/bin/env python3
"""
Educational Data Mining Tools for Physics Assistant
Advanced pattern recognition and insight extraction from educational data,
including learning behavior analysis, performance prediction, and
educational effectiveness measurement.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, silhouette_score
from sklearn.decomposition import PCA
import scipy.stats as stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LearningPattern:
    """Identified learning pattern with characteristics"""
    pattern_id: str
    pattern_type: str  # 'sequential', 'clustered', 'distributed', 'intensive'
    description: str
    frequency: float
    effectiveness_score: float
    associated_concepts: List[str]
    student_characteristics: Dict[str, Any]
    success_indicators: List[str]
    risk_factors: List[str]

@dataclass
class PerformancePrediction:
    """Performance prediction for a student"""
    student_id: str
    prediction_type: str  # 'mastery', 'success_rate', 'completion_time'
    predicted_value: float
    confidence_interval: Tuple[float, float]
    confidence_score: float
    contributing_factors: Dict[str, float]
    intervention_recommendations: List[str]
    prediction_horizon: str  # '1_week', '1_month', '3_months'

@dataclass
class EducationalInsight:
    """Educational insight derived from data mining"""
    insight_id: str
    insight_type: str  # 'trend', 'anomaly', 'correlation', 'recommendation'
    title: str
    description: str
    significance_score: float
    affected_population: str  # 'all_students', 'specific_group', 'individual'
    actionable_recommendations: List[str]
    supporting_evidence: Dict[str, Any]
    temporal_validity: str  # 'immediate', 'short_term', 'long_term'

@dataclass
class StudentCluster:
    """Student cluster with shared characteristics"""
    cluster_id: str
    cluster_name: str
    student_ids: List[str]
    characteristics: Dict[str, float]
    common_patterns: List[str]
    success_strategies: List[str]
    risk_factors: List[str]
    recommended_interventions: List[str]

class EducationalDataMiner:
    """Advanced educational data mining and pattern recognition system"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.student_data_cache = {}
        self.interaction_patterns = []
        self.prediction_models = {}
        self.clustering_models = {}
        
        # Configuration parameters
        self.config = {
            'min_pattern_frequency': 0.05,  # 5% of students
            'significance_threshold': 0.1,
            'prediction_horizon_days': 30,
            'clustering_min_samples': 5,
            'anomaly_detection_threshold': 0.1,
            'temporal_window_days': 90,
            'pattern_stability_days': 14
        }
        
        # Initialize pattern recognition algorithms
        self.pattern_detectors = {
            'sequential_learning': self._detect_sequential_patterns,
            'session_clustering': self._detect_session_patterns,
            'concept_transitions': self._detect_concept_transition_patterns,
            'error_propagation': self._detect_error_propagation_patterns,
            'engagement_cycles': self._detect_engagement_patterns
        }
    
    async def initialize(self):
        """Initialize the educational data mining system"""
        try:
            logger.info("🚀 Initializing Educational Data Mining System")
            
            # Load historical data for pattern analysis
            await self._load_historical_data()
            
            # Initialize machine learning models
            await self._initialize_prediction_models()
            
            # Build student clustering models
            await self._build_clustering_models()
            
            # Detect initial patterns
            await self._detect_initial_patterns()
            
            logger.info("✅ Educational Data Mining System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Educational Data Mining System: {e}")
            return False
    
    async def _load_historical_data(self):
        """Load historical student interaction data"""
        if not self.db_manager:
            self._create_sample_data()
            return
        
        try:
            # Load comprehensive interaction data
            async with self.db_manager.postgres.get_connection() as conn:
                interactions = await conn.fetch("""
                    SELECT i.id, i.user_id, i.agent_type, i.success, i.created_at,
                           i.execution_time_ms, i.metadata, i.request_data, i.response_data,
                           u.username, up.proficiency_score, up.problems_attempted, up.problems_solved
                    FROM interactions i
                    JOIN users u ON i.user_id = u.id
                    LEFT JOIN user_progress up ON i.user_id = up.user_id AND i.agent_type = up.topic
                    WHERE i.created_at >= $1
                    ORDER BY i.user_id, i.created_at
                """, datetime.now() - timedelta(days=self.config['temporal_window_days']))
                
                # Organize data by student
                for interaction in interactions:
                    user_id = str(interaction['user_id'])
                    if user_id not in self.student_data_cache:
                        self.student_data_cache[user_id] = {
                            'interactions': [],
                            'profile': {
                                'username': interaction['username'],
                                'total_interactions': 0,
                                'success_rate': 0.0,
                                'avg_response_time': 0.0,
                                'topics_engaged': set()
                            }
                        }
                    
                    self.student_data_cache[user_id]['interactions'].append({
                        'id': str(interaction['id']),
                        'agent_type': interaction['agent_type'],
                        'success': interaction['success'],
                        'timestamp': interaction['created_at'],
                        'execution_time_ms': interaction['execution_time_ms'],
                        'metadata': interaction['metadata'],
                        'proficiency_score': interaction['proficiency_score']
                    })
                    
                    # Update profile
                    profile = self.student_data_cache[user_id]['profile']
                    profile['total_interactions'] += 1
                    if interaction['agent_type']:
                        profile['topics_engaged'].add(interaction['agent_type'])
                
                # Calculate summary statistics
                for user_id, data in self.student_data_cache.items():
                    interactions = data['interactions']
                    if interactions:
                        successes = sum(1 for i in interactions if i['success'])
                        data['profile']['success_rate'] = successes / len(interactions)
                        
                        response_times = [i['execution_time_ms'] for i in interactions 
                                        if i['execution_time_ms'] is not None]
                        if response_times:
                            data['profile']['avg_response_time'] = np.mean(response_times)
                        
                        data['profile']['topics_engaged'] = list(data['profile']['topics_engaged'])
            
            logger.info(f"📊 Loaded data for {len(self.student_data_cache)} students")
            
        except Exception as e:
            logger.error(f"❌ Failed to load historical data: {e}")
            self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample data for testing"""
        self.student_data_cache = {
            'student_1': {
                'interactions': [
                    {'id': '1', 'agent_type': 'kinematics', 'success': True, 
                     'timestamp': datetime.now() - timedelta(days=1), 'execution_time_ms': 15000},
                    {'id': '2', 'agent_type': 'forces', 'success': False, 
                     'timestamp': datetime.now(), 'execution_time_ms': 25000}
                ],
                'profile': {'username': 'test_student_1', 'total_interactions': 2, 
                          'success_rate': 0.5, 'avg_response_time': 20000, 
                          'topics_engaged': ['kinematics', 'forces']}
            }
        }
    
    async def _initialize_prediction_models(self):
        """Initialize machine learning models for performance prediction"""
        try:
            # Prepare training data
            training_features, training_labels = await self._prepare_training_data()
            
            if len(training_features) > 10:  # Minimum data requirement
                # Initialize classifiers for different prediction types
                self.prediction_models = {
                    'success_predictor': RandomForestClassifier(n_estimators=100, random_state=42),
                    'mastery_predictor': RandomForestClassifier(n_estimators=100, random_state=42),
                    'risk_detector': IsolationForest(contamination=0.1, random_state=42)
                }
                
                # Train success predictor
                if 'success' in training_labels:
                    success_labels = training_labels['success']
                    self.prediction_models['success_predictor'].fit(training_features, success_labels)
                
                logger.info("🤖 Prediction models initialized and trained")
            else:
                logger.warning("⚠️ Insufficient data for model training")
                self.prediction_models = {}
        
        except Exception as e:
            logger.error(f"❌ Failed to initialize prediction models: {e}")
            self.prediction_models = {}
    
    async def _prepare_training_data(self):
        """Prepare training data for machine learning models"""
        features = []
        labels = {'success': [], 'mastery': []}
        
        try:
            for user_id, data in self.student_data_cache.items():
                interactions = data['interactions']
                profile = data['profile']
                
                if len(interactions) < 3:  # Need minimum interactions
                    continue
                
                # Extract features for each interaction context
                for i, interaction in enumerate(interactions):
                    feature_vector = [
                        profile['success_rate'],
                        profile['avg_response_time'] / 1000.0,  # Convert to seconds
                        len(profile['topics_engaged']),
                        i / len(interactions),  # Position in sequence
                        1.0 if interaction['agent_type'] == 'kinematics' else 0.0,
                        1.0 if interaction['agent_type'] == 'forces' else 0.0,
                        1.0 if interaction['agent_type'] == 'energy' else 0.0,
                        # Add more features as needed
                    ]
                    
                    features.append(feature_vector)
                    labels['success'].append(1 if interaction['success'] else 0)
                    
                    # Estimate mastery level
                    proficiency = interaction.get('proficiency_score', 50) / 100.0
                    labels['mastery'].append(1 if proficiency > 0.7 else 0)
            
            return np.array(features), labels
        
        except Exception as e:
            logger.error(f"❌ Failed to prepare training data: {e}")
            return np.array([]), {}
    
    async def _build_clustering_models(self):
        """Build student clustering models"""
        try:
            # Prepare clustering features
            clustering_features, student_ids = await self._prepare_clustering_data()
            
            if len(clustering_features) > 5:  # Minimum students for clustering
                # Standardize features
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(clustering_features)
                
                # K-means clustering
                n_clusters = min(5, len(clustering_features) // 2)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(scaled_features)
                
                # DBSCAN for density-based clustering
                dbscan = DBSCAN(eps=0.5, min_samples=2)
                dbscan_labels = dbscan.fit_predict(scaled_features)
                
                self.clustering_models = {
                    'kmeans': {'model': kmeans, 'scaler': scaler, 'labels': cluster_labels},
                    'dbscan': {'model': dbscan, 'scaler': scaler, 'labels': dbscan_labels},
                    'student_ids': student_ids,
                    'features': clustering_features
                }
                
                logger.info(f"👥 Student clustering models built: {n_clusters} K-means clusters")
            else:
                logger.warning("⚠️ Insufficient students for clustering")
        
        except Exception as e:
            logger.error(f"❌ Failed to build clustering models: {e}")
    
    async def _prepare_clustering_data(self):
        """Prepare data for student clustering"""
        features = []
        student_ids = []
        
        try:
            for user_id, data in self.student_data_cache.items():
                interactions = data['interactions']
                profile = data['profile']
                
                if len(interactions) < 2:
                    continue
                
                # Extract clustering features
                feature_vector = [
                    profile['success_rate'],
                    profile['avg_response_time'] / 1000.0,
                    len(profile['topics_engaged']),
                    profile['total_interactions'],
                    # Session patterns
                    self._calculate_session_regularity(interactions),
                    self._calculate_concept_switching_rate(interactions),
                    self._calculate_improvement_rate(interactions),
                    self._calculate_engagement_consistency(interactions)
                ]
                
                features.append(feature_vector)
                student_ids.append(user_id)
            
            return np.array(features), student_ids
        
        except Exception as e:
            logger.error(f"❌ Failed to prepare clustering data: {e}")
            return np.array([]), []
    
    def _calculate_session_regularity(self, interactions: List[Dict]) -> float:
        """Calculate how regularly a student engages in sessions"""
        if len(interactions) < 3:
            return 0.5
        
        timestamps = [i['timestamp'] for i in interactions]
        timestamps.sort()
        
        # Calculate time intervals between sessions
        intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                    for i in range(len(timestamps) - 1)]
        
        if not intervals:
            return 0.5
        
        # Regularity is inverse of coefficient of variation
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        if mean_interval == 0:
            return 0.5
        
        cv = std_interval / mean_interval
        regularity = max(0.0, min(1.0, 1.0 - cv))
        
        return regularity
    
    def _calculate_concept_switching_rate(self, interactions: List[Dict]) -> float:
        """Calculate how frequently a student switches between concepts"""
        if len(interactions) < 2:
            return 0.0
        
        switches = 0
        for i in range(1, len(interactions)):
            if interactions[i]['agent_type'] != interactions[i-1]['agent_type']:
                switches += 1
        
        return switches / (len(interactions) - 1)
    
    def _calculate_improvement_rate(self, interactions: List[Dict]) -> float:
        """Calculate the rate of improvement over time"""
        if len(interactions) < 5:
            return 0.5
        
        # Use moving average of success rate
        window_size = min(5, len(interactions) // 2)
        early_window = interactions[:window_size]
        late_window = interactions[-window_size:]
        
        early_success = sum(1 for i in early_window if i['success']) / len(early_window)
        late_success = sum(1 for i in late_window if i['success']) / len(late_window)
        
        improvement = late_success - early_success
        return max(0.0, min(1.0, 0.5 + improvement))
    
    def _calculate_engagement_consistency(self, interactions: List[Dict]) -> float:
        """Calculate consistency of engagement over time"""
        if len(interactions) < 3:
            return 0.5
        
        # Group interactions by day
        daily_counts = defaultdict(int)
        for interaction in interactions:
            day = interaction['timestamp'].date()
            daily_counts[day] += 1
        
        if len(daily_counts) < 2:
            return 0.5
        
        # Calculate consistency as inverse of coefficient of variation
        counts = list(daily_counts.values())
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        
        if mean_count == 0:
            return 0.5
        
        cv = std_count / mean_count
        consistency = max(0.0, min(1.0, 1.0 - cv))
        
        return consistency
    
    async def _detect_initial_patterns(self):
        """Detect initial learning patterns from data"""
        try:
            # Run all pattern detection algorithms
            all_patterns = []
            
            for pattern_name, detector_func in self.pattern_detectors.items():
                patterns = await detector_func()
                all_patterns.extend(patterns)
            
            self.interaction_patterns = all_patterns
            logger.info(f"🔍 Detected {len(all_patterns)} learning patterns")
            
        except Exception as e:
            logger.error(f"❌ Failed to detect initial patterns: {e}")
    
    async def _detect_sequential_patterns(self) -> List[LearningPattern]:
        """Detect sequential learning patterns"""
        patterns = []
        
        try:
            # Analyze concept sequences across all students
            concept_sequences = []
            
            for user_id, data in self.student_data_cache.items():
                interactions = data['interactions']
                if len(interactions) >= 3:
                    sequence = [i['agent_type'] for i in interactions if i['agent_type']]
                    if len(sequence) >= 3:
                        concept_sequences.append(sequence)
            
            if not concept_sequences:
                return patterns
            
            # Find common subsequences
            sequence_counter = Counter()
            for sequence in concept_sequences:
                for i in range(len(sequence) - 2):
                    subseq = tuple(sequence[i:i+3])
                    sequence_counter[subseq] += 1
            
            # Identify significant patterns
            total_sequences = len(concept_sequences)
            for subseq, count in sequence_counter.items():
                frequency = count / total_sequences
                if frequency >= self.config['min_pattern_frequency']:
                    # Calculate effectiveness
                    effectiveness = self._calculate_sequence_effectiveness(subseq, concept_sequences)
                    
                    pattern = LearningPattern(
                        pattern_id=f"sequential_{hash(subseq)}",
                        pattern_type='sequential',
                        description=f"Sequential learning pattern: {' → '.join(subseq)}",
                        frequency=frequency,
                        effectiveness_score=effectiveness,
                        associated_concepts=list(subseq),
                        student_characteristics={},
                        success_indicators=[f"Following {subseq[0]} → {subseq[1]} → {subseq[2]} sequence"],
                        risk_factors=[]
                    )
                    patterns.append(pattern)
            
            return patterns
        
        except Exception as e:
            logger.error(f"❌ Failed to detect sequential patterns: {e}")
            return []
    
    def _calculate_sequence_effectiveness(self, sequence: Tuple[str, ...], 
                                        all_sequences: List[List[str]]) -> float:
        """Calculate effectiveness of a learning sequence"""
        try:
            # Find students who followed this sequence
            success_rates = []
            
            for student_sequence in all_sequences:
                # Find occurrences of the pattern in this student's sequence
                for i in range(len(student_sequence) - len(sequence) + 1):
                    subseq = tuple(student_sequence[i:i+len(sequence)])
                    if subseq == sequence:
                        # Look at success rate in subsequent interactions
                        # (This is a simplified calculation)
                        success_rates.append(0.7)  # Placeholder
            
            return np.mean(success_rates) if success_rates else 0.5
        
        except Exception as e:
            logger.error(f"❌ Failed to calculate sequence effectiveness: {e}")
            return 0.5
    
    async def _detect_session_patterns(self) -> List[LearningPattern]:
        """Detect session-based learning patterns"""
        patterns = []
        
        try:
            # Analyze session characteristics
            session_data = []
            
            for user_id, data in self.student_data_cache.items():
                interactions = data['interactions']
                
                # Group interactions into sessions (within 1 hour)
                sessions = self._group_into_sessions(interactions)
                
                for session in sessions:
                    if len(session) >= 2:
                        session_duration = (session[-1]['timestamp'] - session[0]['timestamp']).total_seconds() / 60
                        session_success_rate = sum(1 for i in session if i['success']) / len(session)
                        
                        session_data.append({
                            'duration': session_duration,
                            'interaction_count': len(session),
                            'success_rate': session_success_rate,
                            'concepts_covered': len(set(i['agent_type'] for i in session if i['agent_type']))
                        })
            
            if not session_data:
                return patterns
            
            # Cluster sessions to find patterns
            features = np.array([
                [s['duration'], s['interaction_count'], s['success_rate'], s['concepts_covered']]
                for s in session_data
            ])
            
            if len(features) > 5:
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(features)
                
                kmeans = KMeans(n_clusters=min(3, len(features) // 2), random_state=42)
                cluster_labels = kmeans.fit_predict(scaled_features)
                
                # Analyze each cluster
                for cluster_id in range(kmeans.n_clusters):
                    cluster_mask = cluster_labels == cluster_id
                    cluster_sessions = [session_data[i] for i in range(len(session_data)) if cluster_mask[i]]
                    
                    if len(cluster_sessions) >= 2:
                        avg_duration = np.mean([s['duration'] for s in cluster_sessions])
                        avg_success = np.mean([s['success_rate'] for s in cluster_sessions])
                        
                        pattern = LearningPattern(
                            pattern_id=f"session_cluster_{cluster_id}",
                            pattern_type='session_based',
                            description=f"Session pattern: {avg_duration:.1f}min duration, {avg_success:.2f} success rate",
                            frequency=len(cluster_sessions) / len(session_data),
                            effectiveness_score=avg_success,
                            associated_concepts=[],
                            student_characteristics={'avg_session_duration': avg_duration},
                            success_indicators=[f"Sessions of {avg_duration:.1f} minutes"],
                            risk_factors=[] if avg_success > 0.6 else ['Short session duration', 'Low success rate']
                        )
                        patterns.append(pattern)
            
            return patterns
        
        except Exception as e:
            logger.error(f"❌ Failed to detect session patterns: {e}")
            return []
    
    def _group_into_sessions(self, interactions: List[Dict]) -> List[List[Dict]]:
        """Group interactions into sessions based on time gaps"""
        if not interactions:
            return []
        
        sessions = []
        current_session = [interactions[0]]
        
        for i in range(1, len(interactions)):
            time_gap = (interactions[i]['timestamp'] - interactions[i-1]['timestamp']).total_seconds()
            
            if time_gap <= 3600:  # 1 hour threshold
                current_session.append(interactions[i])
            else:
                sessions.append(current_session)
                current_session = [interactions[i]]
        
        sessions.append(current_session)
        return sessions
    
    async def _detect_concept_transition_patterns(self) -> List[LearningPattern]:
        """Detect patterns in concept transitions"""
        patterns = []
        
        try:
            # Build transition matrix
            transitions = defaultdict(int)
            transition_outcomes = defaultdict(list)
            
            for user_id, data in self.student_data_cache.items():
                interactions = data['interactions']
                
                for i in range(1, len(interactions)):
                    prev_concept = interactions[i-1]['agent_type']
                    curr_concept = interactions[i]['agent_type']
                    
                    if prev_concept and curr_concept and prev_concept != curr_concept:
                        transition = (prev_concept, curr_concept)
                        transitions[transition] += 1
                        transition_outcomes[transition].append(interactions[i]['success'])
            
            # Analyze transition effectiveness
            for transition, count in transitions.items():
                if count >= 3:  # Minimum occurrences
                    outcomes = transition_outcomes[transition]
                    success_rate = sum(outcomes) / len(outcomes)
                    
                    if success_rate >= 0.7 or success_rate <= 0.3:  # Significant pattern
                        pattern = LearningPattern(
                            pattern_id=f"transition_{transition[0]}_{transition[1]}",
                            pattern_type='concept_transition',
                            description=f"Transition from {transition[0]} to {transition[1]}",
                            frequency=count / sum(transitions.values()),
                            effectiveness_score=success_rate,
                            associated_concepts=[transition[0], transition[1]],
                            student_characteristics={},
                            success_indicators=[f"Transitioning from {transition[0]} to {transition[1]}"] if success_rate >= 0.7 else [],
                            risk_factors=[f"Difficult transition from {transition[0]} to {transition[1]}"] if success_rate <= 0.3 else []
                        )
                        patterns.append(pattern)
            
            return patterns
        
        except Exception as e:
            logger.error(f"❌ Failed to detect concept transition patterns: {e}")
            return []
    
    async def _detect_error_propagation_patterns(self) -> List[LearningPattern]:
        """Detect patterns in error propagation"""
        patterns = []
        
        try:
            # Analyze error sequences
            error_sequences = []
            
            for user_id, data in self.student_data_cache.items():
                interactions = data['interactions']
                
                # Find sequences of failures
                current_error_seq = []
                for interaction in interactions:
                    if not interaction['success']:
                        current_error_seq.append(interaction['agent_type'])
                    else:
                        if len(current_error_seq) >= 2:
                            error_sequences.append(current_error_seq.copy())
                        current_error_seq = []
                
                # Add final sequence if it exists
                if len(current_error_seq) >= 2:
                    error_sequences.append(current_error_seq)
            
            if error_sequences:
                # Find common error patterns
                error_pattern_counter = Counter()
                for seq in error_sequences:
                    for i in range(len(seq) - 1):
                        pattern = (seq[i], seq[i+1])
                        error_pattern_counter[pattern] += 1
                
                total_error_sequences = len(error_sequences)
                for pattern, count in error_pattern_counter.items():
                    frequency = count / total_error_sequences
                    if frequency >= 0.1:  # 10% threshold for error patterns
                        error_pattern = LearningPattern(
                            pattern_id=f"error_prop_{pattern[0]}_{pattern[1]}",
                            pattern_type='error_propagation',
                            description=f"Error propagation: {pattern[0]} → {pattern[1]}",
                            frequency=frequency,
                            effectiveness_score=0.0,  # Error patterns have low effectiveness
                            associated_concepts=[pattern[0], pattern[1]],
                            student_characteristics={},
                            success_indicators=[],
                            risk_factors=[f"Sequential failures in {pattern[0]} and {pattern[1]}"]
                        )
                        patterns.append(error_pattern)
            
            return patterns
        
        except Exception as e:
            logger.error(f"❌ Failed to detect error propagation patterns: {e}")
            return []
    
    async def _detect_engagement_patterns(self) -> List[LearningPattern]:
        """Detect engagement patterns"""
        patterns = []
        
        try:
            # Analyze engagement over time
            engagement_profiles = []
            
            for user_id, data in self.student_data_cache.items():
                interactions = data['interactions']
                
                if len(interactions) >= 5:
                    # Calculate engagement metrics over time
                    daily_engagement = defaultdict(int)
                    for interaction in interactions:
                        day = interaction['timestamp'].date()
                        daily_engagement[day] += 1
                    
                    engagement_values = list(daily_engagement.values())
                    if engagement_values:
                        engagement_profiles.append({
                            'mean_daily': np.mean(engagement_values),
                            'std_daily': np.std(engagement_values),
                            'max_daily': max(engagement_values),
                            'consistency': 1.0 - (np.std(engagement_values) / np.mean(engagement_values)) if np.mean(engagement_values) > 0 else 0
                        })
            
            if len(engagement_profiles) > 3:
                # Cluster engagement patterns
                features = np.array([
                    [p['mean_daily'], p['std_daily'], p['max_daily'], p['consistency']]
                    for p in engagement_profiles
                ])
                
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(features)
                
                kmeans = KMeans(n_clusters=min(3, len(features) // 2), random_state=42)
                cluster_labels = kmeans.fit_predict(scaled_features)
                
                # Analyze engagement clusters
                for cluster_id in range(kmeans.n_clusters):
                    cluster_mask = cluster_labels == cluster_id
                    cluster_profiles = [engagement_profiles[i] for i in range(len(engagement_profiles)) if cluster_mask[i]]
                    
                    if len(cluster_profiles) >= 2:
                        avg_consistency = np.mean([p['consistency'] for p in cluster_profiles])
                        avg_daily = np.mean([p['mean_daily'] for p in cluster_profiles])
                        
                        pattern = LearningPattern(
                            pattern_id=f"engagement_{cluster_id}",
                            pattern_type='engagement',
                            description=f"Engagement pattern: {avg_daily:.1f} avg daily interactions, {avg_consistency:.2f} consistency",
                            frequency=len(cluster_profiles) / len(engagement_profiles),
                            effectiveness_score=avg_consistency,
                            associated_concepts=[],
                            student_characteristics={'avg_daily_interactions': avg_daily, 'consistency': avg_consistency},
                            success_indicators=[f"Consistent daily engagement ({avg_daily:.1f} interactions)"] if avg_consistency > 0.7 else [],
                            risk_factors=[f"Inconsistent engagement pattern"] if avg_consistency < 0.4 else []
                        )
                        patterns.append(pattern)
            
            return patterns
        
        except Exception as e:
            logger.error(f"❌ Failed to detect engagement patterns: {e}")
            return []
    
    async def predict_student_performance(self, student_id: str, 
                                        prediction_type: str = 'success_rate',
                                        horizon_days: int = 30) -> PerformancePrediction:
        """Predict student performance using machine learning models"""
        try:
            # Get student data
            if student_id not in self.student_data_cache:
                return PerformancePrediction(
                    student_id=student_id,
                    prediction_type=prediction_type,
                    predicted_value=0.5,
                    confidence_interval=(0.0, 1.0),
                    confidence_score=0.0,
                    contributing_factors={},
                    intervention_recommendations=["Insufficient data for prediction"],
                    prediction_horizon=f"{horizon_days}_days"
                )
            
            student_data = self.student_data_cache[student_id]
            
            # Extract features for prediction
            features = self._extract_prediction_features(student_data)
            
            # Make prediction based on type
            if prediction_type == 'success_rate' and 'success_predictor' in self.prediction_models:
                model = self.prediction_models['success_predictor']
                
                # For probabilistic prediction
                predicted_proba = model.predict_proba(features.reshape(1, -1))[0]
                predicted_value = predicted_proba[1] if len(predicted_proba) > 1 else 0.5
                
                # Calculate confidence interval (simplified)
                confidence_margin = 0.1  # ±10%
                confidence_interval = (
                    max(0.0, predicted_value - confidence_margin),
                    min(1.0, predicted_value + confidence_margin)
                )
                
                # Feature importance as contributing factors
                feature_names = ['success_rate', 'avg_response_time', 'topics_engaged', 'position', 
                               'kinematics', 'forces', 'energy']
                importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else [1.0/len(feature_names)] * len(feature_names)
                
                contributing_factors = dict(zip(feature_names[:len(importances)], importances))
                
                # Generate recommendations
                recommendations = self._generate_performance_recommendations(
                    student_data, predicted_value, contributing_factors
                )
                
                return PerformancePrediction(
                    student_id=student_id,
                    prediction_type=prediction_type,
                    predicted_value=predicted_value,
                    confidence_interval=confidence_interval,
                    confidence_score=0.8,  # Simplified confidence score
                    contributing_factors=contributing_factors,
                    intervention_recommendations=recommendations,
                    prediction_horizon=f"{horizon_days}_days"
                )
            
            else:
                # Fallback prediction based on historical performance
                historical_success = student_data['profile']['success_rate']
                
                return PerformancePrediction(
                    student_id=student_id,
                    prediction_type=prediction_type,
                    predicted_value=historical_success,
                    confidence_interval=(historical_success * 0.8, min(1.0, historical_success * 1.2)),
                    confidence_score=0.6,
                    contributing_factors={'historical_performance': 1.0},
                    intervention_recommendations=self._generate_basic_recommendations(historical_success),
                    prediction_horizon=f"{horizon_days}_days"
                )
        
        except Exception as e:
            logger.error(f"❌ Failed to predict student performance: {e}")
            return PerformancePrediction(
                student_id=student_id,
                prediction_type=prediction_type,
                predicted_value=0.5,
                confidence_interval=(0.0, 1.0),
                confidence_score=0.0,
                contributing_factors={},
                intervention_recommendations=["Prediction failed"],
                prediction_horizon=f"{horizon_days}_days"
            )
    
    def _extract_prediction_features(self, student_data: Dict) -> np.ndarray:
        """Extract features for performance prediction"""
        try:
            profile = student_data['profile']
            interactions = student_data['interactions']
            
            # Basic features
            features = [
                profile['success_rate'],
                profile['avg_response_time'] / 1000.0,
                len(profile['topics_engaged']),
                len(interactions),
                # Recent performance
                self._calculate_recent_performance(interactions),
                self._calculate_improvement_trend(interactions),
                self._calculate_concept_diversity(interactions),
                # Add topic-specific features
                1.0 if 'kinematics' in profile['topics_engaged'] else 0.0,
                1.0 if 'forces' in profile['topics_engaged'] else 0.0,
                1.0 if 'energy' in profile['topics_engaged'] else 0.0
            ]
            
            return np.array(features, dtype=float)
        
        except Exception as e:
            logger.error(f"❌ Failed to extract prediction features: {e}")
            return np.array([0.5] * 7, dtype=float)
    
    def _calculate_recent_performance(self, interactions: List[Dict]) -> float:
        """Calculate performance in recent interactions"""
        if not interactions:
            return 0.5
        
        recent_count = min(10, len(interactions))
        recent_interactions = interactions[-recent_count:]
        
        successes = sum(1 for i in recent_interactions if i['success'])
        return successes / len(recent_interactions)
    
    def _calculate_improvement_trend(self, interactions: List[Dict]) -> float:
        """Calculate improvement trend over time"""
        if len(interactions) < 4:
            return 0.0
        
        mid_point = len(interactions) // 2
        early_performance = sum(1 for i in interactions[:mid_point] if i['success']) / mid_point
        late_performance = sum(1 for i in interactions[mid_point:] if i['success']) / (len(interactions) - mid_point)
        
        return late_performance - early_performance
    
    def _calculate_concept_diversity(self, interactions: List[Dict]) -> float:
        """Calculate diversity of concepts engaged with"""
        if not interactions:
            return 0.0
        
        concepts = set(i['agent_type'] for i in interactions if i['agent_type'])
        return len(concepts)
    
    def _generate_performance_recommendations(self, student_data: Dict, predicted_performance: float,
                                            contributing_factors: Dict[str, float]) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        try:
            profile = student_data['profile']
            
            # Performance-based recommendations
            if predicted_performance < 0.6:
                recommendations.append("Consider reviewing prerequisite concepts")
                recommendations.append("Increase practice frequency")
            
            # Factor-based recommendations
            top_factors = sorted(contributing_factors.items(), key=lambda x: x[1], reverse=True)[:3]
            
            for factor, importance in top_factors:
                if factor == 'avg_response_time' and profile['avg_response_time'] > 20000:
                    recommendations.append("Focus on reducing response time - break down complex problems")
                elif factor == 'success_rate' and profile['success_rate'] < 0.7:
                    recommendations.append("Work on improving accuracy through targeted practice")
                elif factor == 'topics_engaged' and len(profile['topics_engaged']) < 3:
                    recommendations.append("Explore additional physics topics to build comprehensive understanding")
            
            return recommendations[:5]  # Limit to top 5 recommendations
        
        except Exception as e:
            logger.error(f"❌ Failed to generate performance recommendations: {e}")
            return ["Continue current learning approach"]
    
    def _generate_basic_recommendations(self, success_rate: float) -> List[str]:
        """Generate basic recommendations based on success rate"""
        if success_rate >= 0.8:
            return ["Continue excellent progress", "Consider exploring advanced topics"]
        elif success_rate >= 0.6:
            return ["Good progress - maintain current pace", "Focus on areas of difficulty"]
        else:
            return ["Increase practice frequency", "Review fundamental concepts", "Consider additional support"]
    
    async def identify_student_clusters(self) -> List[StudentCluster]:
        """Identify and analyze student clusters"""
        try:
            if not self.clustering_models or 'kmeans' not in self.clustering_models:
                return []
            
            clustering_data = self.clustering_models
            kmeans_labels = clustering_data['kmeans']['labels']
            student_ids = clustering_data['student_ids']
            features = clustering_data['features']
            
            clusters = []
            
            # Analyze each cluster
            for cluster_id in range(clustering_data['kmeans']['model'].n_clusters):
                cluster_mask = kmeans_labels == cluster_id
                cluster_student_ids = [student_ids[i] for i in range(len(student_ids)) if cluster_mask[i]]
                cluster_features = features[cluster_mask]
                
                if len(cluster_student_ids) > 0:
                    # Calculate cluster characteristics
                    avg_features = np.mean(cluster_features, axis=0)
                    
                    characteristics = {
                        'avg_success_rate': avg_features[0],
                        'avg_response_time': avg_features[1],
                        'avg_topics_engaged': avg_features[2],
                        'avg_total_interactions': avg_features[3],
                        'session_regularity': avg_features[4] if len(avg_features) > 4 else 0.5,
                        'concept_switching_rate': avg_features[5] if len(avg_features) > 5 else 0.5,
                        'improvement_rate': avg_features[6] if len(avg_features) > 6 else 0.5,
                        'engagement_consistency': avg_features[7] if len(avg_features) > 7 else 0.5
                    }
                    
                    # Generate cluster insights
                    cluster_name = self._generate_cluster_name(characteristics)
                    common_patterns = self._identify_cluster_patterns(cluster_student_ids)
                    success_strategies = self._identify_success_strategies(characteristics)
                    risk_factors = self._identify_risk_factors(characteristics)
                    interventions = self._recommend_cluster_interventions(characteristics)
                    
                    cluster = StudentCluster(
                        cluster_id=f"cluster_{cluster_id}",
                        cluster_name=cluster_name,
                        student_ids=cluster_student_ids,
                        characteristics=characteristics,
                        common_patterns=common_patterns,
                        success_strategies=success_strategies,
                        risk_factors=risk_factors,
                        recommended_interventions=interventions
                    )
                    clusters.append(cluster)
            
            return clusters
        
        except Exception as e:
            logger.error(f"❌ Failed to identify student clusters: {e}")
            return []
    
    def _generate_cluster_name(self, characteristics: Dict[str, float]) -> str:
        """Generate descriptive name for student cluster"""
        success_rate = characteristics['avg_success_rate']
        engagement = characteristics['engagement_consistency']
        
        if success_rate >= 0.8 and engagement >= 0.7:
            return "High Achievers"
        elif success_rate >= 0.6 and engagement >= 0.6:
            return "Steady Learners"
        elif success_rate < 0.5 or engagement < 0.4:
            return "At-Risk Students"
        elif characteristics['concept_switching_rate'] > 0.7:
            return "Exploratory Learners"
        else:
            return "Developing Students"
    
    def _identify_cluster_patterns(self, student_ids: List[str]) -> List[str]:
        """Identify common patterns within a cluster"""
        patterns = []
        
        # Analyze patterns specific to this cluster
        # This is a simplified implementation
        if len(student_ids) > 2:
            patterns.append("Consistent learning schedule")
            patterns.append("Similar concept progression")
        
        return patterns
    
    def _identify_success_strategies(self, characteristics: Dict[str, float]) -> List[str]:
        """Identify successful strategies for this cluster"""
        strategies = []
        
        if characteristics['session_regularity'] > 0.7:
            strategies.append("Regular study sessions")
        
        if characteristics['improvement_rate'] > 0.1:
            strategies.append("Continuous improvement approach")
        
        if characteristics['avg_success_rate'] > 0.7:
            strategies.append("High accuracy focus")
        
        return strategies
    
    def _identify_risk_factors(self, characteristics: Dict[str, float]) -> List[str]:
        """Identify risk factors for this cluster"""
        risk_factors = []
        
        if characteristics['avg_success_rate'] < 0.5:
            risk_factors.append("Low success rate")
        
        if characteristics['engagement_consistency'] < 0.4:
            risk_factors.append("Inconsistent engagement")
        
        if characteristics['concept_switching_rate'] > 0.8:
            risk_factors.append("Excessive concept switching")
        
        return risk_factors
    
    def _recommend_cluster_interventions(self, characteristics: Dict[str, float]) -> List[str]:
        """Recommend interventions for this cluster"""
        interventions = []
        
        if characteristics['avg_success_rate'] < 0.6:
            interventions.append("Provide additional scaffolding and support")
            interventions.append("Implement mastery-based progression")
        
        if characteristics['engagement_consistency'] < 0.5:
            interventions.append("Implement gamification elements")
            interventions.append("Provide regular progress feedback")
        
        if characteristics['avg_response_time'] > 25000:  # 25 seconds
            interventions.append("Provide hints and guided practice")
            interventions.append("Break down complex problems")
        
        return interventions
    
    async def generate_educational_insights(self, timeframe_days: int = 30) -> List[EducationalInsight]:
        """Generate educational insights from data analysis"""
        insights = []
        
        try:
            # Trend analysis
            trend_insights = await self._analyze_trends(timeframe_days)
            insights.extend(trend_insights)
            
            # Anomaly detection
            anomaly_insights = await self._detect_anomalies()
            insights.extend(anomaly_insights)
            
            # Correlation analysis
            correlation_insights = await self._analyze_correlations()
            insights.extend(correlation_insights)
            
            # Performance insights
            performance_insights = await self._analyze_performance_patterns()
            insights.extend(performance_insights)
            
            # Sort by significance
            insights.sort(key=lambda x: x.significance_score, reverse=True)
            
            return insights[:10]  # Return top 10 insights
        
        except Exception as e:
            logger.error(f"❌ Failed to generate educational insights: {e}")
            return []
    
    async def _analyze_trends(self, timeframe_days: int) -> List[EducationalInsight]:
        """Analyze trending patterns in student data"""
        insights = []
        
        try:
            # Calculate overall success rate trend
            all_interactions = []
            for user_data in self.student_data_cache.values():
                all_interactions.extend(user_data['interactions'])
            
            if len(all_interactions) > 10:
                # Sort by timestamp
                all_interactions.sort(key=lambda x: x['timestamp'])
                
                # Split into early and recent periods
                mid_point = len(all_interactions) // 2
                early_success = sum(1 for i in all_interactions[:mid_point] if i['success']) / mid_point
                recent_success = sum(1 for i in all_interactions[mid_point:] if i['success']) / (len(all_interactions) - mid_point)
                
                trend = recent_success - early_success
                
                if abs(trend) > 0.1:  # Significant trend
                    trend_direction = "improving" if trend > 0 else "declining"
                    
                    insight = EducationalInsight(
                        insight_id=f"trend_success_rate_{timeframe_days}d",
                        insight_type='trend',
                        title=f"Overall Success Rate {trend_direction.title()}",
                        description=f"Student success rates are {trend_direction} by {abs(trend):.1%} over the analysis period",
                        significance_score=abs(trend) * 2,  # Scale significance
                        affected_population='all_students',
                        actionable_recommendations=self._generate_trend_recommendations(trend_direction, trend),
                        supporting_evidence={'trend_magnitude': trend, 'sample_size': len(all_interactions)},
                        temporal_validity='short_term'
                    )
                    insights.append(insight)
            
            return insights
        
        except Exception as e:
            logger.error(f"❌ Failed to analyze trends: {e}")
            return []
    
    def _generate_trend_recommendations(self, trend_direction: str, magnitude: float) -> List[str]:
        """Generate recommendations based on trend analysis"""
        if trend_direction == "improving":
            return [
                "Continue current successful approaches",
                "Share effective strategies with other students",
                "Consider introducing more challenging content"
            ]
        else:
            return [
                "Investigate factors contributing to decline",
                "Implement additional student support measures",
                "Review and adjust difficulty levels"
            ]
    
    async def _detect_anomalies(self) -> List[EducationalInsight]:
        """Detect anomalous patterns in student behavior"""
        insights = []
        
        try:
            # Use isolation forest for anomaly detection if available
            if 'risk_detector' in self.prediction_models:
                # Prepare data for anomaly detection
                student_features = []
                student_ids = []
                
                for user_id, data in self.student_data_cache.items():
                    features = self._extract_prediction_features(data)
                    student_features.append(features)
                    student_ids.append(user_id)
                
                if len(student_features) > 5:
                    features_array = np.array(student_features)
                    anomaly_scores = self.prediction_models['risk_detector'].decision_function(features_array)
                    anomalies = self.prediction_models['risk_detector'].predict(features_array)
                    
                    anomalous_students = [student_ids[i] for i in range(len(student_ids)) if anomalies[i] == -1]
                    
                    if anomalous_students:
                        insight = EducationalInsight(
                            insight_id="anomaly_detection_students",
                            insight_type='anomaly',
                            title="Unusual Learning Patterns Detected",
                            description=f"Detected {len(anomalous_students)} students with unusual learning patterns",
                            significance_score=len(anomalous_students) / len(student_ids),
                            affected_population='specific_group',
                            actionable_recommendations=[
                                "Investigate individual student circumstances",
                                "Provide personalized support interventions",
                                "Monitor progress closely"
                            ],
                            supporting_evidence={'anomalous_students': len(anomalous_students), 'total_students': len(student_ids)},
                            temporal_validity='immediate'
                        )
                        insights.append(insight)
            
            return insights
        
        except Exception as e:
            logger.error(f"❌ Failed to detect anomalies: {e}")
            return []
    
    async def _analyze_correlations(self) -> List[EducationalInsight]:
        """Analyze correlations between different factors"""
        insights = []
        
        try:
            # Analyze correlation between response time and success rate
            response_times = []
            success_rates = []
            
            for user_data in self.student_data_cache.values():
                profile = user_data['profile']
                if profile['avg_response_time'] > 0:
                    response_times.append(profile['avg_response_time'])
                    success_rates.append(profile['success_rate'])
            
            if len(response_times) > 3:
                correlation, p_value = stats.pearsonr(response_times, success_rates)
                
                if abs(correlation) > 0.5 and p_value < 0.05:  # Significant correlation
                    direction = "positive" if correlation > 0 else "negative"
                    
                    insight = EducationalInsight(
                        insight_id="correlation_response_time_success",
                        insight_type='correlation',
                        title=f"{direction.title()} Correlation: Response Time vs Success Rate",
                        description=f"Found {direction} correlation ({correlation:.2f}) between response time and success rate",
                        significance_score=abs(correlation),
                        affected_population='all_students',
                        actionable_recommendations=self._generate_correlation_recommendations(correlation),
                        supporting_evidence={'correlation': correlation, 'p_value': p_value},
                        temporal_validity='long_term'
                    )
                    insights.append(insight)
            
            return insights
        
        except Exception as e:
            logger.error(f"❌ Failed to analyze correlations: {e}")
            return []
    
    def _generate_correlation_recommendations(self, correlation: float) -> List[str]:
        """Generate recommendations based on correlation analysis"""
        if correlation < -0.5:  # Negative correlation between response time and success
            return [
                "Encourage students to take more time on problems",
                "Provide guidance on problem-solving strategies",
                "Emphasize accuracy over speed"
            ]
        elif correlation > 0.5:  # Positive correlation
            return [
                "Help students develop efficiency in problem-solving",
                "Provide time management strategies",
                "Balance speed and accuracy in practice"
            ]
        else:
            return ["Monitor response time patterns for individual students"]
    
    async def _analyze_performance_patterns(self) -> List[EducationalInsight]:
        """Analyze overall performance patterns"""
        insights = []
        
        try:
            # Analyze concept-specific performance
            concept_performance = defaultdict(list)
            
            for user_data in self.student_data_cache.values():
                for interaction in user_data['interactions']:
                    if interaction['agent_type']:
                        concept_performance[interaction['agent_type']].append(interaction['success'])
            
            # Find concepts with concerning performance
            concerning_concepts = []
            for concept, outcomes in concept_performance.items():
                if len(outcomes) >= 5:  # Minimum sample size
                    success_rate = sum(outcomes) / len(outcomes)
                    if success_rate < 0.5:  # Below 50% success rate
                        concerning_concepts.append((concept, success_rate, len(outcomes)))
            
            if concerning_concepts:
                # Sort by combination of low success rate and high interaction count
                concerning_concepts.sort(key=lambda x: (x[1], -x[2]))
                worst_concept = concerning_concepts[0]
                
                insight = EducationalInsight(
                    insight_id=f"performance_concern_{worst_concept[0]}",
                    insight_type='recommendation',
                    title=f"Low Performance in {worst_concept[0].title()}",
                    description=f"Students showing {worst_concept[1]:.1%} success rate in {worst_concept[0]} ({worst_concept[2]} interactions)",
                    significance_score=1.0 - worst_concept[1],  # Higher significance for lower success
                    affected_population='all_students',
                    actionable_recommendations=[
                        f"Review {worst_concept[0]} curriculum content",
                        f"Provide additional support materials for {worst_concept[0]}",
                        f"Consider prerequisite review for {worst_concept[0]}",
                        "Implement peer tutoring for struggling concepts"
                    ],
                    supporting_evidence={
                        'concept': worst_concept[0],
                        'success_rate': worst_concept[1],
                        'interaction_count': worst_concept[2]
                    },
                    temporal_validity='short_term'
                )
                insights.append(insight)
            
            return insights
        
        except Exception as e:
            logger.error(f"❌ Failed to analyze performance patterns: {e}")
            return []

# Example usage and testing
async def test_educational_data_mining():
    """Test function for educational data mining"""
    try:
        logger.info("🧪 Testing Educational Data Mining System")
        
        # Initialize data miner
        miner = EducationalDataMiner()
        await miner.initialize()
        
        # Test pattern detection
        if miner.interaction_patterns:
            logger.info(f"✅ Detected {len(miner.interaction_patterns)} learning patterns")
            for pattern in miner.interaction_patterns[:3]:
                logger.info(f"  - {pattern.pattern_type}: {pattern.description}")
        
        # Test performance prediction
        prediction = await miner.predict_student_performance("student_1", "success_rate")
        logger.info(f"✅ Performance prediction: {prediction.predicted_value:.2f} (confidence: {prediction.confidence_score:.2f})")
        
        # Test clustering
        clusters = await miner.identify_student_clusters()
        if clusters:
            logger.info(f"✅ Identified {len(clusters)} student clusters")
            for cluster in clusters:
                logger.info(f"  - {cluster.cluster_name}: {len(cluster.student_ids)} students")
        
        # Test insights generation
        insights = await miner.generate_educational_insights()
        if insights:
            logger.info(f"✅ Generated {len(insights)} educational insights")
            for insight in insights[:3]:
                logger.info(f"  - {insight.insight_type}: {insight.title}")
        
        logger.info("✅ Educational Data Mining test completed")
        
    except Exception as e:
        logger.error(f"❌ Educational Data Mining test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_educational_data_mining())