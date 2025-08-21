#!/usr/bin/env python3
"""
Content Effectiveness Analytics Engine for Physics Assistant
Analyzes effectiveness of physics concepts, problems, teaching materials, and learning paths.
Provides recommendations for content optimization and personalized learning experiences.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from textstat import flesch_reading_ease, flesch_kincaid_grade

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ContentMetrics:
    """Metrics for content effectiveness analysis"""
    content_id: str
    content_type: str  # 'concept', 'problem', 'explanation', 'example'
    engagement_score: float
    learning_effectiveness: float
    difficulty_rating: float
    completion_rate: float
    success_rate: float
    time_to_mastery: float
    student_satisfaction: float
    interaction_count: int
    unique_students: int
    average_attempts: float
    help_requests: int
    hint_usage_rate: float
    error_patterns: Dict[str, int]
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class LearningPathEffectiveness:
    """Effectiveness analysis for learning paths"""
    path_id: str
    path_name: str
    concepts_sequence: List[str]
    completion_rate: float
    average_duration: float
    success_rate: float
    drop_off_points: List[Tuple[str, float]]  # (concept, drop_off_rate)
    optimization_opportunities: List[str]
    recommended_modifications: List[str]
    student_feedback_score: float
    path_efficiency_score: float

@dataclass
class ConceptDifficultyAnalysis:
    """Analysis of concept difficulty and learning progression"""
    concept_name: str
    inherent_difficulty: float
    student_perceived_difficulty: float
    prerequisite_coverage: float
    mastery_time_distribution: Dict[str, float]  # percentiles
    common_misconceptions: List[str]
    effective_teaching_approaches: List[str]
    recommended_prerequisites: List[str]
    optimal_practice_amount: int

@dataclass
class ContentRecommendation:
    """Recommendation for content optimization"""
    content_id: str
    recommendation_type: str  # 'improve', 'replace', 'supplement', 'reorder'
    priority: str  # 'high', 'medium', 'low'
    rationale: str
    suggested_changes: List[str]
    expected_impact: Dict[str, float]  # metric -> expected improvement
    implementation_effort: str  # 'low', 'medium', 'high'
    success_probability: float

@dataclass
class EngagementPattern:
    """Student engagement patterns with content"""
    pattern_id: str
    pattern_type: str  # 'high_engagement', 'rapid_disengagement', 'struggle_persistence'
    student_characteristics: Dict[str, Any]
    content_characteristics: Dict[str, Any]
    typical_progression: List[str]
    intervention_points: List[str]
    success_indicators: List[str]

class ContentEffectivenessEngine:
    """Advanced content effectiveness analytics engine"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        
        # Content registry
        self.content_metrics: Dict[str, ContentMetrics] = {}
        self.learning_paths: Dict[str, LearningPathEffectiveness] = {}
        self.concept_analyses: Dict[str, ConceptDifficultyAnalysis] = {}
        
        # Analysis configuration
        self.analysis_config = {
            'minimum_interactions': 20,
            'engagement_weight_factors': {
                'session_duration': 0.3,
                'return_frequency': 0.25,
                'help_seeking': 0.2,
                'completion_rate': 0.25
            },
            'effectiveness_thresholds': {
                'high': 0.8,
                'medium': 0.6,
                'low': 0.4
            },
            'difficulty_calibration': {
                'very_easy': 0.2,
                'easy': 0.4,
                'medium': 0.6,
                'hard': 0.8,
                'very_hard': 1.0
            }
        }
        
        # Content taxonomy
        self.content_taxonomy = {
            'physics_concepts': {
                'kinematics': ['position', 'velocity', 'acceleration', 'motion_graphs'],
                'forces': ['newtons_laws', 'friction', 'normal_force', 'tension'],
                'energy': ['kinetic_energy', 'potential_energy', 'conservation', 'work'],
                'momentum': ['linear_momentum', 'collisions', 'impulse', 'conservation'],
                'angular_motion': ['rotational_kinematics', 'torque', 'angular_momentum'],
                'waves': ['wave_properties', 'interference', 'standing_waves'],
                'thermodynamics': ['heat', 'temperature', 'entropy', 'gas_laws']
            },
            'problem_types': {
                'conceptual': ['definition', 'explanation', 'comparison'],
                'computational': ['calculation', 'formula_application', 'unit_conversion'],
                'graphical': ['graph_interpretation', 'sketch_drawing', 'data_analysis'],
                'experimental': ['procedure_design', 'data_collection', 'error_analysis']
            }
        }
    
    async def initialize(self):
        """Initialize the content effectiveness engine"""
        try:
            logger.info("🚀 Initializing Content Effectiveness Engine")
            
            # Create content analysis tables
            await self._create_content_tables()
            
            # Load existing content metrics
            await self._load_content_metrics()
            
            # Initialize content taxonomy in knowledge graph
            await self._initialize_content_taxonomy()
            
            logger.info("✅ Content Effectiveness Engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Content Effectiveness Engine: {e}")
            return False
    
    async def _create_content_tables(self):
        """Create database tables for content analytics"""
        try:
            if not self.db_manager:
                return
            
            async with self.db_manager.postgres.get_connection() as conn:
                # Content metrics table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS content_metrics (
                        content_id VARCHAR(100) PRIMARY KEY,
                        content_type VARCHAR(50) NOT NULL,
                        engagement_score FLOAT DEFAULT 0.0,
                        learning_effectiveness FLOAT DEFAULT 0.0,
                        difficulty_rating FLOAT DEFAULT 0.0,
                        completion_rate FLOAT DEFAULT 0.0,
                        success_rate FLOAT DEFAULT 0.0,
                        time_to_mastery FLOAT DEFAULT 0.0,
                        student_satisfaction FLOAT DEFAULT 0.0,
                        interaction_count INTEGER DEFAULT 0,
                        unique_students INTEGER DEFAULT 0,
                        average_attempts FLOAT DEFAULT 0.0,
                        help_requests INTEGER DEFAULT 0,
                        hint_usage_rate FLOAT DEFAULT 0.0,
                        error_patterns JSONB DEFAULT '{}',
                        last_updated TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                # Learning path effectiveness table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS learning_path_effectiveness (
                        path_id VARCHAR(100) PRIMARY KEY,
                        path_name VARCHAR(200) NOT NULL,
                        concepts_sequence JSONB NOT NULL,
                        completion_rate FLOAT DEFAULT 0.0,
                        average_duration FLOAT DEFAULT 0.0,
                        success_rate FLOAT DEFAULT 0.0,
                        drop_off_points JSONB DEFAULT '[]',
                        optimization_opportunities JSONB DEFAULT '[]',
                        recommended_modifications JSONB DEFAULT '[]',
                        student_feedback_score FLOAT DEFAULT 0.0,
                        path_efficiency_score FLOAT DEFAULT 0.0,
                        last_analyzed TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                # Content recommendations table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS content_recommendations (
                        recommendation_id SERIAL PRIMARY KEY,
                        content_id VARCHAR(100) NOT NULL,
                        recommendation_type VARCHAR(50) NOT NULL,
                        priority VARCHAR(20) NOT NULL,
                        rationale TEXT NOT NULL,
                        suggested_changes JSONB DEFAULT '[]',
                        expected_impact JSONB DEFAULT '{}',
                        implementation_effort VARCHAR(20) DEFAULT 'medium',
                        success_probability FLOAT DEFAULT 0.5,
                        created_at TIMESTAMP DEFAULT NOW(),
                        is_implemented BOOLEAN DEFAULT FALSE
                    )
                """)
                
                logger.info("✅ Content analytics tables created")
        
        except Exception as e:
            logger.error(f"❌ Failed to create content tables: {e}")
    
    async def _load_content_metrics(self):
        """Load existing content metrics from database"""
        try:
            if not self.db_manager:
                return
            
            async with self.db_manager.postgres.get_connection() as conn:
                metrics = await conn.fetch("SELECT * FROM content_metrics")
                
                for metric_row in metrics:
                    content_metric = ContentMetrics(
                        content_id=metric_row['content_id'],
                        content_type=metric_row['content_type'],
                        engagement_score=metric_row['engagement_score'],
                        learning_effectiveness=metric_row['learning_effectiveness'],
                        difficulty_rating=metric_row['difficulty_rating'],
                        completion_rate=metric_row['completion_rate'],
                        success_rate=metric_row['success_rate'],
                        time_to_mastery=metric_row['time_to_mastery'],
                        student_satisfaction=metric_row['student_satisfaction'],
                        interaction_count=metric_row['interaction_count'],
                        unique_students=metric_row['unique_students'],
                        average_attempts=metric_row['average_attempts'],
                        help_requests=metric_row['help_requests'],
                        hint_usage_rate=metric_row['hint_usage_rate'],
                        error_patterns=metric_row['error_patterns'],
                        last_updated=metric_row['last_updated']
                    )
                    self.content_metrics[content_metric.content_id] = content_metric
                
                logger.info(f"📊 Loaded {len(self.content_metrics)} content metrics")
        
        except Exception as e:
            logger.error(f"❌ Failed to load content metrics: {e}")
    
    async def _initialize_content_taxonomy(self):
        """Initialize content taxonomy in knowledge graph"""
        try:
            if not self.db_manager:
                return
            
            # Add content taxonomy to Neo4j knowledge graph
            for domain, concepts in self.content_taxonomy['physics_concepts'].items():
                # Create domain nodes
                await self.db_manager.neo4j.run_query("""
                    MERGE (d:Domain {name: $domain})
                    SET d.type = 'physics_domain'
                """, domain=domain)
                
                # Create concept nodes and relationships
                for concept in concepts:
                    await self.db_manager.neo4j.run_query("""
                        MERGE (c:Concept {name: $concept})
                        SET c.domain = $domain, c.type = 'physics_concept'
                        WITH c
                        MATCH (d:Domain {name: $domain})
                        MERGE (d)-[:CONTAINS]->(c)
                    """, concept=concept, domain=domain)
            
            logger.info("✅ Content taxonomy initialized in knowledge graph")
        
        except Exception as e:
            logger.error(f"❌ Failed to initialize content taxonomy: {e}")
    
    async def analyze_content_effectiveness(self, content_id: str, content_type: str) -> ContentMetrics:
        """Analyze effectiveness of specific content"""
        try:
            logger.info(f"📊 Analyzing effectiveness of content: {content_id}")
            
            # Collect interaction data for content
            interaction_data = await self._collect_content_interactions(content_id, content_type)
            
            if not interaction_data:
                logger.warning(f"No interaction data found for content: {content_id}")
                return ContentMetrics(
                    content_id=content_id,
                    content_type=content_type,
                    engagement_score=0.0,
                    learning_effectiveness=0.0,
                    difficulty_rating=0.5,
                    completion_rate=0.0,
                    success_rate=0.0,
                    time_to_mastery=0.0,
                    student_satisfaction=0.0,
                    interaction_count=0,
                    unique_students=0,
                    average_attempts=0.0,
                    help_requests=0,
                    hint_usage_rate=0.0,
                    error_patterns={}
                )
            
            # Calculate basic metrics
            basic_metrics = self._calculate_basic_content_metrics(interaction_data)
            
            # Calculate engagement score
            engagement_score = self._calculate_engagement_score(interaction_data)
            
            # Calculate learning effectiveness
            learning_effectiveness = self._calculate_learning_effectiveness(interaction_data)
            
            # Calculate difficulty rating
            difficulty_rating = self._calculate_content_difficulty(interaction_data)
            
            # Analyze error patterns
            error_patterns = self._analyze_error_patterns(interaction_data)
            
            # Calculate student satisfaction (based on behavior indicators)
            student_satisfaction = self._estimate_student_satisfaction(interaction_data)
            
            content_metrics = ContentMetrics(
                content_id=content_id,
                content_type=content_type,
                engagement_score=engagement_score,
                learning_effectiveness=learning_effectiveness,
                difficulty_rating=difficulty_rating,
                completion_rate=basic_metrics['completion_rate'],
                success_rate=basic_metrics['success_rate'],
                time_to_mastery=basic_metrics['time_to_mastery'],
                student_satisfaction=student_satisfaction,
                interaction_count=basic_metrics['interaction_count'],
                unique_students=basic_metrics['unique_students'],
                average_attempts=basic_metrics['average_attempts'],
                help_requests=basic_metrics['help_requests'],
                hint_usage_rate=basic_metrics['hint_usage_rate'],
                error_patterns=error_patterns
            )
            
            # Store metrics
            self.content_metrics[content_id] = content_metrics
            await self._save_content_metrics(content_metrics)
            
            logger.info(f"✅ Content effectiveness analysis completed for {content_id}")
            return content_metrics
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze content effectiveness for {content_id}: {e}")
            raise
    
    async def _collect_content_interactions(self, content_id: str, content_type: str) -> List[Dict[str, Any]]:
        """Collect all interactions related to specific content"""
        try:
            if not self.db_manager:
                return []
            
            async with self.db_manager.postgres.get_connection() as conn:
                # Query based on content type
                if content_type == 'concept':
                    # Get interactions with specific agent (concept)
                    interactions = await conn.fetch("""
                        SELECT * FROM interactions 
                        WHERE agent_type = $1 AND created_at >= NOW() - INTERVAL '90 days'
                        ORDER BY created_at ASC
                    """, content_id)
                else:
                    # Get interactions with specific content mentioned in metadata
                    interactions = await conn.fetch("""
                        SELECT * FROM interactions 
                        WHERE metadata::text ILIKE $1 AND created_at >= NOW() - INTERVAL '90 days'
                        ORDER BY created_at ASC
                    """, f'%{content_id}%')
                
                return [dict(row) for row in interactions]
        
        except Exception as e:
            logger.error(f"❌ Failed to collect content interactions: {e}")
            return []
    
    def _calculate_basic_content_metrics(self, interaction_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate basic content metrics from interaction data"""
        try:
            if not interaction_data:
                return {
                    'completion_rate': 0.0,
                    'success_rate': 0.0,
                    'time_to_mastery': 0.0,
                    'interaction_count': 0,
                    'unique_students': 0,
                    'average_attempts': 0.0,
                    'help_requests': 0,
                    'hint_usage_rate': 0.0
                }
            
            df = pd.DataFrame(interaction_data)
            
            # Basic counts
            interaction_count = len(df)
            unique_students = df['user_id'].nunique()
            
            # Success metrics
            success_rate = df['success'].mean() if 'success' in df.columns else 0.0
            
            # Completion analysis (simplified - based on successful interactions)
            completed_students = df[df['success'] == True]['user_id'].nunique() if 'success' in df.columns else 0
            completion_rate = completed_students / unique_students if unique_students > 0 else 0.0
            
            # Attempts analysis
            student_attempts = df.groupby('user_id').size()
            average_attempts = student_attempts.mean()
            
            # Help-seeking behavior
            help_requests = 0
            hint_usage_count = 0
            
            for _, row in df.iterrows():
                if row.get('metadata'):
                    try:
                        metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
                        if metadata.get('help_requested'):
                            help_requests += 1
                        if metadata.get('hint_used'):
                            hint_usage_count += 1
                    except:
                        pass
            
            hint_usage_rate = hint_usage_count / interaction_count if interaction_count > 0 else 0.0
            
            # Time to mastery (simplified - time until first success for each student)
            mastery_times = []
            for student_id in df['user_id'].unique():
                student_interactions = df[df['user_id'] == student_id].sort_values('created_at')
                first_success = student_interactions[student_interactions['success'] == True].head(1)
                
                if not first_success.empty:
                    first_interaction = student_interactions.iloc[0]['created_at']
                    success_time = first_success.iloc[0]['created_at']
                    time_diff = (success_time - first_interaction).total_seconds() / 3600  # hours
                    mastery_times.append(time_diff)
            
            time_to_mastery = np.mean(mastery_times) if mastery_times else 0.0
            
            return {
                'completion_rate': completion_rate,
                'success_rate': success_rate,
                'time_to_mastery': time_to_mastery,
                'interaction_count': interaction_count,
                'unique_students': unique_students,
                'average_attempts': average_attempts,
                'help_requests': help_requests,
                'hint_usage_rate': hint_usage_rate
            }
        
        except Exception as e:
            logger.error(f"❌ Failed to calculate basic content metrics: {e}")
            return {}
    
    def _calculate_engagement_score(self, interaction_data: List[Dict[str, Any]]) -> float:
        """Calculate content engagement score based on student behavior"""
        try:
            if not interaction_data:
                return 0.0
            
            df = pd.DataFrame(interaction_data)
            
            # Engagement factors
            engagement_factors = {}
            
            # 1. Session duration (time spent on content)
            if 'execution_time_ms' in df.columns:
                avg_session_time = df['execution_time_ms'].mean() / 1000 / 60  # minutes
                # Normalize to 0-1 scale (assuming 5 minutes is high engagement)
                engagement_factors['session_duration'] = min(avg_session_time / 5.0, 1.0)
            else:
                engagement_factors['session_duration'] = 0.5
            
            # 2. Return frequency (students coming back)
            daily_active_students = df.groupby(df['created_at'].dt.date)['user_id'].nunique()
            unique_students = df['user_id'].nunique()
            return_rate = daily_active_students.mean() / max(unique_students, 1)
            engagement_factors['return_frequency'] = min(return_rate, 1.0)
            
            # 3. Help-seeking behavior (moderate help-seeking indicates engagement)
            help_requests = 0
            for _, row in df.iterrows():
                if row.get('metadata'):
                    try:
                        metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
                        if metadata.get('help_requested') or metadata.get('hint_used'):
                            help_requests += 1
                    except:
                        pass
            
            help_rate = help_requests / len(df)
            # Optimal help rate is around 20-30%
            if help_rate <= 0.3:
                engagement_factors['help_seeking'] = help_rate / 0.3
            else:
                engagement_factors['help_seeking'] = max(0, 1.0 - (help_rate - 0.3) / 0.7)
            
            # 4. Completion behavior
            completion_rate = self._calculate_basic_content_metrics(interaction_data)['completion_rate']
            engagement_factors['completion_rate'] = completion_rate
            
            # Calculate weighted engagement score
            weights = self.analysis_config['engagement_weight_factors']
            engagement_score = sum(
                engagement_factors.get(factor, 0.5) * weight
                for factor, weight in weights.items()
            )
            
            return min(max(engagement_score, 0.0), 1.0)
        
        except Exception as e:
            logger.error(f"❌ Failed to calculate engagement score: {e}")
            return 0.5
    
    def _calculate_learning_effectiveness(self, interaction_data: List[Dict[str, Any]]) -> float:
        """Calculate learning effectiveness based on student progress"""
        try:
            if not interaction_data:
                return 0.0
            
            df = pd.DataFrame(interaction_data)
            
            # Learning effectiveness factors
            effectiveness_factors = []
            
            # 1. Success rate improvement over time
            if 'success' in df.columns and len(df) >= 10:
                # Divide interactions into early and late periods
                mid_point = len(df) // 2
                early_success = df.iloc[:mid_point]['success'].mean()
                late_success = df.iloc[mid_point:]['success'].mean()
                improvement = late_success - early_success
                effectiveness_factors.append(min(max(improvement + 0.5, 0), 1))  # Normalize around 0.5
            
            # 2. Time efficiency (decreasing time to solve problems)
            if 'execution_time_ms' in df.columns and len(df) >= 10:
                mid_point = len(df) // 2
                early_time = df.iloc[:mid_point]['execution_time_ms'].mean()
                late_time = df.iloc[mid_point:]['execution_time_ms'].mean()
                
                if early_time > 0:
                    time_improvement = (early_time - late_time) / early_time
                    effectiveness_factors.append(min(max(time_improvement, 0), 1))
            
            # 3. Reduced help-seeking over time
            early_help = 0
            late_help = 0
            mid_point = len(df) // 2
            
            for i, row in df.iterrows():
                if row.get('metadata'):
                    try:
                        metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
                        if metadata.get('help_requested') or metadata.get('hint_used'):
                            if i < mid_point:
                                early_help += 1
                            else:
                                late_help += 1
                    except:
                        pass
            
            if mid_point > 0:
                early_help_rate = early_help / mid_point
                late_help_rate = late_help / (len(df) - mid_point)
                help_reduction = early_help_rate - late_help_rate
                effectiveness_factors.append(min(max(help_reduction, 0), 1))
            
            # 4. Overall success rate
            success_rate = df['success'].mean() if 'success' in df.columns else 0.0
            effectiveness_factors.append(success_rate)
            
            # Calculate average effectiveness
            if effectiveness_factors:
                return np.mean(effectiveness_factors)
            else:
                return 0.5
        
        except Exception as e:
            logger.error(f"❌ Failed to calculate learning effectiveness: {e}")
            return 0.5
    
    def _calculate_content_difficulty(self, interaction_data: List[Dict[str, Any]]) -> float:
        """Calculate perceived difficulty of content based on student performance"""
        try:
            if not interaction_data:
                return 0.5
            
            df = pd.DataFrame(interaction_data)
            
            # Difficulty indicators
            difficulty_indicators = []
            
            # 1. Success rate (lower success = higher difficulty)
            if 'success' in df.columns:
                success_rate = df['success'].mean()
                difficulty_indicators.append(1.0 - success_rate)
            
            # 2. Time to complete (longer time = higher difficulty)
            if 'execution_time_ms' in df.columns:
                avg_time = df['execution_time_ms'].mean() / 1000 / 60  # minutes
                # Normalize to 0-1 scale (10 minutes = very difficult)
                time_difficulty = min(avg_time / 10.0, 1.0)
                difficulty_indicators.append(time_difficulty)
            
            # 3. Help-seeking frequency (more help = higher difficulty)
            help_requests = 0
            for _, row in df.iterrows():
                if row.get('metadata'):
                    try:
                        metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
                        if metadata.get('help_requested') or metadata.get('hint_used'):
                            help_requests += 1
                    except:
                        pass
            
            help_rate = help_requests / len(df)
            difficulty_indicators.append(min(help_rate / 0.5, 1.0))  # 50% help rate = max difficulty
            
            # 4. Attempt frequency (more attempts = higher difficulty)
            student_attempts = df.groupby('user_id').size()
            avg_attempts = student_attempts.mean()
            # Normalize to 0-1 scale (5 attempts = very difficult)
            attempt_difficulty = min((avg_attempts - 1) / 4.0, 1.0)
            difficulty_indicators.append(max(attempt_difficulty, 0))
            
            # Calculate average difficulty
            if difficulty_indicators:
                return np.mean(difficulty_indicators)
            else:
                return 0.5
        
        except Exception as e:
            logger.error(f"❌ Failed to calculate content difficulty: {e}")
            return 0.5
    
    def _analyze_error_patterns(self, interaction_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze common error patterns in content interactions"""
        try:
            error_patterns = Counter()
            
            for row in interaction_data:
                if not row.get('success', True) and row.get('metadata'):
                    try:
                        metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
                        
                        # Extract error information
                        error_type = metadata.get('error_type')
                        if error_type:
                            error_patterns[error_type] += 1
                        
                        # Extract problem-specific errors
                        problem_errors = metadata.get('problem_errors', [])
                        for error in problem_errors:
                            error_patterns[error] += 1
                        
                        # Extract conceptual errors
                        conceptual_errors = metadata.get('conceptual_errors', [])
                        for error in conceptual_errors:
                            error_patterns[f"conceptual_{error}"] += 1
                    
                    except:
                        pass
            
            return dict(error_patterns.most_common(10))  # Top 10 error patterns
        
        except Exception as e:
            logger.error(f"❌ Failed to analyze error patterns: {e}")
            return {}
    
    def _estimate_student_satisfaction(self, interaction_data: List[Dict[str, Any]]) -> float:
        """Estimate student satisfaction based on behavioral indicators"""
        try:
            if not interaction_data:
                return 0.5
            
            df = pd.DataFrame(interaction_data)
            satisfaction_indicators = []
            
            # 1. Session completion (did students finish sessions?)
            # Proxy: low drop-off rate in sessions
            session_completion_rate = 1.0  # Simplified - assume completion
            satisfaction_indicators.append(session_completion_rate)
            
            # 2. Return behavior (do students come back?)
            unique_students = df['user_id'].nunique()
            returning_students = df.groupby('user_id').size()
            return_rate = (returning_students > 1).sum() / unique_students if unique_students > 0 else 0
            satisfaction_indicators.append(return_rate)
            
            # 3. Balanced help-seeking (not too frustrated, not too easy)
            help_requests = 0
            for _, row in df.iterrows():
                if row.get('metadata'):
                    try:
                        metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
                        if metadata.get('help_requested'):
                            help_requests += 1
                    except:
                        pass
            
            help_rate = help_requests / len(df)
            # Optimal help rate for satisfaction is around 15-25%
            if 0.15 <= help_rate <= 0.25:
                help_satisfaction = 1.0
            elif help_rate < 0.15:
                help_satisfaction = help_rate / 0.15  # Too easy
            else:
                help_satisfaction = max(0, 1.0 - (help_rate - 0.25) / 0.25)  # Too hard
            
            satisfaction_indicators.append(help_satisfaction)
            
            # 4. Success rate (successful experiences increase satisfaction)
            success_rate = df['success'].mean() if 'success' in df.columns else 0.5
            satisfaction_indicators.append(success_rate)
            
            # Calculate average satisfaction
            return np.mean(satisfaction_indicators)
        
        except Exception as e:
            logger.error(f"❌ Failed to estimate student satisfaction: {e}")
            return 0.5
    
    async def analyze_learning_path_effectiveness(self, path_id: str, concepts_sequence: List[str]) -> LearningPathEffectiveness:
        """Analyze effectiveness of a learning path"""
        try:
            logger.info(f"📚 Analyzing learning path effectiveness: {path_id}")
            
            # Collect data for each concept in the path
            path_data = []
            for concept in concepts_sequence:
                concept_data = await self._collect_content_interactions(concept, 'concept')
                path_data.append((concept, concept_data))
            
            # Analyze completion rates through the path
            completion_rates = []
            drop_off_points = []
            
            for i, (concept, concept_data) in enumerate(path_data):
                if concept_data:
                    df = pd.DataFrame(concept_data)
                    unique_students = df['user_id'].nunique()
                    
                    if i == 0:
                        initial_students = unique_students
                    
                    completion_rate = unique_students / initial_students if initial_students > 0 else 0
                    completion_rates.append(completion_rate)
                    
                    # Check for significant drop-offs
                    if i > 0 and completion_rate < completion_rates[i-1] * 0.8:  # 20% drop
                        drop_off_rate = completion_rates[i-1] - completion_rate
                        drop_off_points.append((concept, drop_off_rate))
                else:
                    completion_rates.append(0.0)
            
            # Calculate overall path metrics
            overall_completion_rate = completion_rates[-1] if completion_rates else 0.0
            
            # Calculate average duration through path
            durations = []
            for student_id in set().union(*[pd.DataFrame(data)['user_id'].unique() if data else [] for _, data in path_data]):
                student_start = None
                student_end = None
                
                for concept, concept_data in path_data:
                    if concept_data:
                        df = pd.DataFrame(concept_data)
                        student_interactions = df[df['user_id'] == student_id]
                        
                        if not student_interactions.empty:
                            if student_start is None:
                                student_start = student_interactions['created_at'].min()
                            student_end = student_interactions['created_at'].max()
                
                if student_start and student_end:
                    duration = (student_end - student_start).total_seconds() / 3600  # hours
                    durations.append(duration)
            
            average_duration = np.mean(durations) if durations else 0.0
            
            # Calculate success rate through path
            path_success_rates = []
            for concept, concept_data in path_data:
                if concept_data:
                    df = pd.DataFrame(concept_data)
                    success_rate = df['success'].mean() if 'success' in df.columns else 0.0
                    path_success_rates.append(success_rate)
            
            overall_success_rate = np.mean(path_success_rates) if path_success_rates else 0.0
            
            # Generate optimization opportunities
            optimization_opportunities = self._identify_path_optimization_opportunities(
                concepts_sequence, completion_rates, path_success_rates, drop_off_points
            )
            
            # Generate recommended modifications
            recommended_modifications = self._generate_path_modifications(
                concepts_sequence, optimization_opportunities
            )
            
            # Calculate path efficiency score
            path_efficiency_score = self._calculate_path_efficiency(
                overall_completion_rate, overall_success_rate, average_duration, len(concepts_sequence)
            )
            
            path_effectiveness = LearningPathEffectiveness(
                path_id=path_id,
                path_name=f"Learning Path {path_id}",
                concepts_sequence=concepts_sequence,
                completion_rate=overall_completion_rate,
                average_duration=average_duration,
                success_rate=overall_success_rate,
                drop_off_points=drop_off_points,
                optimization_opportunities=optimization_opportunities,
                recommended_modifications=recommended_modifications,
                student_feedback_score=0.7,  # Placeholder - would need actual feedback
                path_efficiency_score=path_efficiency_score
            )
            
            # Store results
            self.learning_paths[path_id] = path_effectiveness
            await self._save_learning_path_effectiveness(path_effectiveness)
            
            logger.info(f"✅ Learning path effectiveness analysis completed for {path_id}")
            return path_effectiveness
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze learning path effectiveness: {e}")
            raise
    
    def _identify_path_optimization_opportunities(self, concepts_sequence: List[str], 
                                                completion_rates: List[float],
                                                success_rates: List[float],
                                                drop_off_points: List[Tuple[str, float]]) -> List[str]:
        """Identify optimization opportunities for learning path"""
        opportunities = []
        
        try:
            # Check for concepts with low success rates
            for i, (concept, success_rate) in enumerate(zip(concepts_sequence, success_rates)):
                if success_rate < 0.6:
                    opportunities.append(f"Review content difficulty for {concept} (success rate: {success_rate:.2%})")
            
            # Check for significant drop-off points
            for concept, drop_rate in drop_off_points:
                opportunities.append(f"Address student drop-off at {concept} (drop rate: {drop_rate:.2%})")
            
            # Check for completion rate decline
            if completion_rates:
                final_rate = completion_rates[-1]
                if final_rate < 0.5:
                    opportunities.append(f"Improve overall path completion rate ({final_rate:.2%})")
            
            # Check for prerequisite gaps
            for i in range(1, len(concepts_sequence)):
                if i < len(success_rates) and success_rates[i] < success_rates[i-1] * 0.8:
                    opportunities.append(f"Check prerequisites between {concepts_sequence[i-1]} and {concepts_sequence[i]}")
            
        except Exception as e:
            logger.error(f"❌ Failed to identify optimization opportunities: {e}")
        
        return opportunities
    
    def _generate_path_modifications(self, concepts_sequence: List[str], 
                                   opportunities: List[str]) -> List[str]:
        """Generate specific modification recommendations for learning path"""
        modifications = []
        
        try:
            # Based on opportunities, suggest specific modifications
            for opportunity in opportunities:
                if "content difficulty" in opportunity:
                    modifications.append("Add prerequisite review materials before difficult concepts")
                    modifications.append("Include more practice problems with scaffolding")
                
                elif "drop-off" in opportunity:
                    modifications.append("Add motivational elements and progress indicators")
                    modifications.append("Provide alternative explanation approaches")
                
                elif "completion rate" in opportunity:
                    modifications.append("Break long concepts into smaller, manageable chunks")
                    modifications.append("Add checkpoint assessments to maintain engagement")
                
                elif "prerequisites" in opportunity:
                    modifications.append("Add prerequisite check and review modules")
                    modifications.append("Reorder concepts based on dependency analysis")
            
            # General improvements
            if len(modifications) == 0:
                modifications.append("Monitor student progress and provide personalized feedback")
                modifications.append("Add optional enrichment activities for advanced students")
        
        except Exception as e:
            logger.error(f"❌ Failed to generate path modifications: {e}")
        
        return modifications
    
    def _calculate_path_efficiency(self, completion_rate: float, success_rate: float, 
                                 average_duration: float, num_concepts: int) -> float:
        """Calculate overall efficiency score for learning path"""
        try:
            # Efficiency factors
            efficiency_factors = []
            
            # 1. Completion efficiency
            efficiency_factors.append(completion_rate)
            
            # 2. Learning efficiency (success rate)
            efficiency_factors.append(success_rate)
            
            # 3. Time efficiency (normalized by number of concepts)
            expected_time_per_concept = 2.0  # 2 hours per concept
            expected_total_time = num_concepts * expected_time_per_concept
            
            if average_duration > 0:
                time_efficiency = min(expected_total_time / average_duration, 1.0)
            else:
                time_efficiency = 0.5
            
            efficiency_factors.append(time_efficiency)
            
            # Calculate weighted average
            weights = [0.4, 0.4, 0.2]  # Completion and success are more important than time
            efficiency_score = sum(factor * weight for factor, weight in zip(efficiency_factors, weights))
            
            return min(max(efficiency_score, 0.0), 1.0)
        
        except Exception as e:
            logger.error(f"❌ Failed to calculate path efficiency: {e}")
            return 0.5
    
    async def generate_content_recommendations(self, content_id: str) -> List[ContentRecommendation]:
        """Generate recommendations for content improvement"""
        try:
            logger.info(f"💡 Generating content recommendations for: {content_id}")
            
            if content_id not in self.content_metrics:
                await self.analyze_content_effectiveness(content_id, 'concept')
            
            metrics = self.content_metrics.get(content_id)
            if not metrics:
                return []
            
            recommendations = []
            
            # Analyze each metric and generate recommendations
            
            # 1. Low engagement recommendations
            if metrics.engagement_score < self.analysis_config['effectiveness_thresholds']['medium']:
                rec = ContentRecommendation(
                    content_id=content_id,
                    recommendation_type='improve',
                    priority='high' if metrics.engagement_score < 0.4 else 'medium',
                    rationale=f"Low engagement score ({metrics.engagement_score:.2f})",
                    suggested_changes=[
                        "Add interactive elements and visualizations",
                        "Include real-world examples and applications",
                        "Break content into smaller, digestible chunks",
                        "Add gamification elements"
                    ],
                    expected_impact={'engagement_score': 0.3, 'completion_rate': 0.2},
                    implementation_effort='medium',
                    success_probability=0.7
                )
                recommendations.append(rec)
            
            # 2. Low learning effectiveness recommendations
            if metrics.learning_effectiveness < self.analysis_config['effectiveness_thresholds']['medium']:
                rec = ContentRecommendation(
                    content_id=content_id,
                    recommendation_type='improve',
                    priority='high',
                    rationale=f"Low learning effectiveness ({metrics.learning_effectiveness:.2f})",
                    suggested_changes=[
                        "Improve prerequisite coverage",
                        "Add more practice opportunities",
                        "Include step-by-step problem solving guides",
                        "Provide immediate feedback mechanisms"
                    ],
                    expected_impact={'learning_effectiveness': 0.4, 'success_rate': 0.25},
                    implementation_effort='high',
                    success_probability=0.8
                )
                recommendations.append(rec)
            
            # 3. Difficulty calibration recommendations
            if metrics.difficulty_rating > 0.8:
                rec = ContentRecommendation(
                    content_id=content_id,
                    recommendation_type='improve',
                    priority='medium',
                    rationale=f"Content appears too difficult ({metrics.difficulty_rating:.2f})",
                    suggested_changes=[
                        "Add prerequisite review sections",
                        "Provide more scaffolding and hints",
                        "Include conceptual explanations before procedural content",
                        "Add worked examples"
                    ],
                    expected_impact={'difficulty_rating': -0.2, 'success_rate': 0.2},
                    implementation_effort='medium',
                    success_probability=0.75
                )
                recommendations.append(rec)
            elif metrics.difficulty_rating < 0.3:
                rec = ContentRecommendation(
                    content_id=content_id,
                    recommendation_type='supplement',
                    priority='low',
                    rationale=f"Content may be too easy ({metrics.difficulty_rating:.2f})",
                    suggested_changes=[
                        "Add extension problems for advanced students",
                        "Include application challenges",
                        "Provide optional deeper exploration topics"
                    ],
                    expected_impact={'engagement_score': 0.15, 'difficulty_rating': 0.1},
                    implementation_effort='low',
                    success_probability=0.6
                )
                recommendations.append(rec)
            
            # 4. Error pattern based recommendations
            if metrics.error_patterns:
                most_common_error = max(metrics.error_patterns.items(), key=lambda x: x[1])
                rec = ContentRecommendation(
                    content_id=content_id,
                    recommendation_type='improve',
                    priority='medium',
                    rationale=f"Common error pattern detected: {most_common_error[0]}",
                    suggested_changes=[
                        f"Address common misconception: {most_common_error[0]}",
                        "Add targeted practice for error-prone areas",
                        "Provide clear explanations of correct approaches"
                    ],
                    expected_impact={'success_rate': 0.15, 'error_reduction': 0.3},
                    implementation_effort='medium',
                    success_probability=0.7
                )
                recommendations.append(rec)
            
            # 5. Low completion rate recommendations
            if metrics.completion_rate < 0.6:
                rec = ContentRecommendation(
                    content_id=content_id,
                    recommendation_type='improve',
                    priority='high',
                    rationale=f"Low completion rate ({metrics.completion_rate:.2%})",
                    suggested_changes=[
                        "Shorten content length",
                        "Add progress indicators",
                        "Include motivational elements",
                        "Provide clearer learning objectives"
                    ],
                    expected_impact={'completion_rate': 0.25, 'engagement_score': 0.2},
                    implementation_effort='medium',
                    success_probability=0.65
                )
                recommendations.append(rec)
            
            # Save recommendations
            for rec in recommendations:
                await self._save_content_recommendation(rec)
            
            logger.info(f"✅ Generated {len(recommendations)} content recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Failed to generate content recommendations: {e}")
            return []
    
    async def _save_content_metrics(self, metrics: ContentMetrics):
        """Save content metrics to database"""
        try:
            if not self.db_manager:
                return
            
            async with self.db_manager.postgres.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO content_metrics 
                    (content_id, content_type, engagement_score, learning_effectiveness, 
                     difficulty_rating, completion_rate, success_rate, time_to_mastery,
                     student_satisfaction, interaction_count, unique_students, 
                     average_attempts, help_requests, hint_usage_rate, error_patterns)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                    ON CONFLICT (content_id) DO UPDATE SET
                    engagement_score = EXCLUDED.engagement_score,
                    learning_effectiveness = EXCLUDED.learning_effectiveness,
                    difficulty_rating = EXCLUDED.difficulty_rating,
                    completion_rate = EXCLUDED.completion_rate,
                    success_rate = EXCLUDED.success_rate,
                    time_to_mastery = EXCLUDED.time_to_mastery,
                    student_satisfaction = EXCLUDED.student_satisfaction,
                    interaction_count = EXCLUDED.interaction_count,
                    unique_students = EXCLUDED.unique_students,
                    average_attempts = EXCLUDED.average_attempts,
                    help_requests = EXCLUDED.help_requests,
                    hint_usage_rate = EXCLUDED.hint_usage_rate,
                    error_patterns = EXCLUDED.error_patterns,
                    last_updated = NOW()
                """, 
                metrics.content_id, metrics.content_type, metrics.engagement_score,
                metrics.learning_effectiveness, metrics.difficulty_rating, 
                metrics.completion_rate, metrics.success_rate, metrics.time_to_mastery,
                metrics.student_satisfaction, metrics.interaction_count, 
                metrics.unique_students, metrics.average_attempts, metrics.help_requests,
                metrics.hint_usage_rate, json.dumps(metrics.error_patterns))
        
        except Exception as e:
            logger.error(f"❌ Failed to save content metrics: {e}")
    
    async def _save_learning_path_effectiveness(self, path_effectiveness: LearningPathEffectiveness):
        """Save learning path effectiveness to database"""
        try:
            if not self.db_manager:
                return
            
            async with self.db_manager.postgres.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO learning_path_effectiveness
                    (path_id, path_name, concepts_sequence, completion_rate, average_duration,
                     success_rate, drop_off_points, optimization_opportunities,
                     recommended_modifications, student_feedback_score, path_efficiency_score)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (path_id) DO UPDATE SET
                    completion_rate = EXCLUDED.completion_rate,
                    average_duration = EXCLUDED.average_duration,
                    success_rate = EXCLUDED.success_rate,
                    drop_off_points = EXCLUDED.drop_off_points,
                    optimization_opportunities = EXCLUDED.optimization_opportunities,
                    recommended_modifications = EXCLUDED.recommended_modifications,
                    student_feedback_score = EXCLUDED.student_feedback_score,
                    path_efficiency_score = EXCLUDED.path_efficiency_score,
                    last_analyzed = NOW()
                """,
                path_effectiveness.path_id, path_effectiveness.path_name,
                json.dumps(path_effectiveness.concepts_sequence),
                path_effectiveness.completion_rate, path_effectiveness.average_duration,
                path_effectiveness.success_rate, json.dumps(path_effectiveness.drop_off_points),
                json.dumps(path_effectiveness.optimization_opportunities),
                json.dumps(path_effectiveness.recommended_modifications),
                path_effectiveness.student_feedback_score, path_effectiveness.path_efficiency_score)
        
        except Exception as e:
            logger.error(f"❌ Failed to save learning path effectiveness: {e}")
    
    async def _save_content_recommendation(self, recommendation: ContentRecommendation):
        """Save content recommendation to database"""
        try:
            if not self.db_manager:
                return
            
            async with self.db_manager.postgres.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO content_recommendations
                    (content_id, recommendation_type, priority, rationale, suggested_changes,
                     expected_impact, implementation_effort, success_probability)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                recommendation.content_id, recommendation.recommendation_type,
                recommendation.priority, recommendation.rationale,
                json.dumps(recommendation.suggested_changes),
                json.dumps(recommendation.expected_impact),
                recommendation.implementation_effort, recommendation.success_probability)
        
        except Exception as e:
            logger.error(f"❌ Failed to save content recommendation: {e}")

# Main testing function
async def test_content_effectiveness():
    """Test content effectiveness engine"""
    try:
        logger.info("🧪 Testing Content Effectiveness Engine")
        
        engine = ContentEffectivenessEngine()
        await engine.initialize()
        
        # Test content metrics structure
        sample_metrics = ContentMetrics(
            content_id="kinematics_basics",
            content_type="concept",
            engagement_score=0.75,
            learning_effectiveness=0.68,
            difficulty_rating=0.55,
            completion_rate=0.82,
            success_rate=0.71,
            time_to_mastery=3.5,
            student_satisfaction=0.73,
            interaction_count=450,
            unique_students=85,
            average_attempts=2.3,
            help_requests=67,
            hint_usage_rate=0.18,
            error_patterns={'unit_conversion': 15, 'sign_error': 12}
        )
        
        logger.info(f"✅ Sample content metrics: {sample_metrics.content_id}")
        
        # Test recommendation structure
        sample_recommendation = ContentRecommendation(
            content_id="kinematics_basics",
            recommendation_type="improve",
            priority="medium",
            rationale="Moderate engagement score suggests room for improvement",
            suggested_changes=["Add interactive visualizations", "Include real-world examples"],
            expected_impact={'engagement_score': 0.2, 'success_rate': 0.15},
            implementation_effort="medium",
            success_probability=0.75
        )
        
        logger.info(f"✅ Sample recommendation: {sample_recommendation.recommendation_type}")
        
        logger.info("✅ Content Effectiveness Engine test completed")
        
    except Exception as e:
        logger.error(f"❌ Content Effectiveness test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_content_effectiveness())