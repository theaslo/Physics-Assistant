#!/usr/bin/env python3
"""
Learning Analytics Calculation Engine for Physics Assistant
Provides comprehensive analytics for student learning patterns, progress tracking,
concept mastery detection, and personalized learning recommendations.
"""

import asyncio
import json
import logging
import math
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StudentProfile:
    """Student learning profile with comprehensive metrics"""
    user_id: str
    current_level: str = "beginner"
    learning_velocity: float = 0.0
    engagement_score: float = 0.0
    concept_mastery: Dict[str, float] = field(default_factory=dict)
    struggling_concepts: List[str] = field(default_factory=list)
    strong_concepts: List[str] = field(default_factory=list)
    learning_style: str = "mixed"
    preferred_difficulty: str = "adaptive"
    session_patterns: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class ConceptMastery:
    """Concept mastery assessment with detailed metrics"""
    concept_name: str
    mastery_level: float  # 0.0 to 1.0
    confidence_level: float  # 0.0 to 1.0
    time_to_mastery: Optional[float]  # in hours
    attempts_count: int = 0
    success_rate: float = 0.0
    error_patterns: List[str] = field(default_factory=list)
    prerequisites_mastered: bool = False
    last_interaction: datetime = field(default_factory=datetime.now)

@dataclass
class LearningPath:
    """Optimized learning path between concepts"""
    start_concept: str
    target_concept: str
    path_concepts: List[str]
    estimated_time: float  # in hours
    difficulty_progression: List[float]
    success_probability: float
    prerequisite_gaps: List[str] = field(default_factory=list)

@dataclass
class InteractionPattern:
    """Pattern analysis for student interactions"""
    session_duration_avg: float
    questions_per_session: float
    help_seeking_frequency: float
    error_recovery_time: float
    concept_switching_rate: float
    difficulty_preference: str

class LearningAnalyticsEngine:
    """Core learning analytics calculation engine"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.student_profiles: Dict[str, StudentProfile] = {}
        self.concept_graph = None
        self.interaction_cache: Dict[str, List[Dict]] = defaultdict(list)
        self.mastery_threshold = 0.7
        self.confidence_threshold = 0.8
        
        # Analytics configuration
        self.config = {
            'mastery_window_days': 7,
            'engagement_decay_rate': 0.1,
            'learning_velocity_alpha': 0.3,
            'concept_similarity_threshold': 0.6,
            'intervention_threshold': 0.4,
            'min_interactions_for_mastery': 5
        }
    
    async def initialize(self):
        """Initialize the analytics engine with graph data"""
        try:
            logger.info("🚀 Initializing Learning Analytics Engine")
            
            # Load concept graph from Neo4j
            await self._load_concept_graph()
            
            # Initialize student profiles from database
            await self._initialize_student_profiles()
            
            logger.info("✅ Learning Analytics Engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Learning Analytics Engine: {e}")
            return False
    
    async def _load_concept_graph(self):
        """Load physics concept graph from Neo4j"""
        if not self.db_manager:
            self.concept_graph = nx.DiGraph()
            return
        
        try:
            # Get all concepts and relationships
            concepts_query = """
            MATCH (c:Concept)
            RETURN c.name as name, c.category as category, 
                   c.difficulty as difficulty, c.description as description
            """
            
            relationships_query = """
            MATCH (c1:Concept)-[r]->(c2:Concept)
            RETURN c1.name as source, c2.name as target, 
                   type(r) as relationship_type, 
                   coalesce(r.weight, 1.0) as weight
            """
            
            concepts = await self.db_manager.neo4j.run_query(concepts_query)
            relationships = await self.db_manager.neo4j.run_query(relationships_query)
            
            # Build NetworkX graph
            self.concept_graph = nx.DiGraph()
            
            # Add concept nodes
            for concept in concepts:
                self.concept_graph.add_node(
                    concept['name'],
                    category=concept['category'],
                    difficulty=concept.get('difficulty', 1.0),
                    description=concept['description']
                )
            
            # Add relationships
            for rel in relationships:
                self.concept_graph.add_edge(
                    rel['source'],
                    rel['target'],
                    relationship_type=rel['relationship_type'],
                    weight=rel['weight']
                )
            
            logger.info(f"📊 Loaded concept graph: {len(concepts)} concepts, {len(relationships)} relationships")
            
        except Exception as e:
            logger.error(f"❌ Failed to load concept graph: {e}")
            self.concept_graph = nx.DiGraph()
    
    async def _initialize_student_profiles(self):
        """Initialize student profiles from database"""
        if not self.db_manager:
            return
        
        try:
            # Get all active users
            async with self.db_manager.postgres.get_connection() as conn:
                users = await conn.fetch("""
                    SELECT id, username, created_at, last_login
                    FROM users 
                    WHERE is_active = TRUE
                """)
            
            for user in users:
                user_id = str(user['id'])
                
                # Create initial profile
                profile = StudentProfile(
                    user_id=user_id,
                    current_level="beginner",
                    learning_velocity=0.0,
                    engagement_score=0.0
                )
                
                # Load existing progress if available
                await self._load_user_progress(profile)
                
                self.student_profiles[user_id] = profile
            
            logger.info(f"👥 Initialized profiles for {len(self.student_profiles)} students")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize student profiles: {e}")
    
    async def _load_user_progress(self, profile: StudentProfile):
        """Load existing user progress and calculate initial metrics"""
        if not self.db_manager:
            return
        
        try:
            async with self.db_manager.postgres.get_connection() as conn:
                # Get user progress data
                progress_data = await conn.fetch("""
                    SELECT topic, problems_attempted, problems_solved, 
                           total_interaction_time, proficiency_score
                    FROM user_progress 
                    WHERE user_id = $1
                """, profile.user_id)
                
                # Calculate concept mastery from progress
                for progress in progress_data:
                    topic = progress['topic']
                    if progress['problems_attempted'] > 0:
                        success_rate = progress['problems_solved'] / progress['problems_attempted']
                        profile.concept_mastery[topic] = min(success_rate, progress['proficiency_score'] / 100.0)
                
                # Get recent interactions for pattern analysis
                interactions = await conn.fetch("""
                    SELECT agent_type, created_at, success, metadata, execution_time_ms
                    FROM interactions 
                    WHERE user_id = $1 AND created_at >= $2
                    ORDER BY created_at DESC
                    LIMIT 100
                """, profile.user_id, datetime.now() - timedelta(days=30))
                
                # Analyze interaction patterns
                if interactions:
                    await self._analyze_interaction_patterns(profile, interactions)
        
        except Exception as e:
            logger.error(f"❌ Failed to load user progress for {profile.user_id}: {e}")
    
    async def _analyze_interaction_patterns(self, profile: StudentProfile, interactions: List):
        """Analyze interaction patterns for student profiling"""
        try:
            # Calculate session-based metrics
            sessions = defaultdict(list)
            for interaction in interactions:
                session_date = interaction['created_at'].date()
                sessions[session_date].append(interaction)
            
            session_durations = []
            questions_per_session = []
            success_rates = []
            
            for session_interactions in sessions.values():
                if len(session_interactions) > 1:
                    duration = (session_interactions[0]['created_at'] - 
                              session_interactions[-1]['created_at']).total_seconds() / 3600
                    session_durations.append(duration)
                    questions_per_session.append(len(session_interactions))
                    
                    successes = sum(1 for i in session_interactions if i['success'])
                    success_rates.append(successes / len(session_interactions))
            
            # Update profile metrics
            if session_durations:
                profile.session_patterns = {
                    'avg_duration': np.mean(session_durations),
                    'avg_questions': np.mean(questions_per_session),
                    'avg_success_rate': np.mean(success_rates),
                    'session_count': len(sessions)
                }
                
                # Calculate engagement score (0-1 scale)
                engagement_factors = [
                    min(np.mean(session_durations) / 0.5, 1.0),  # Normalize to 30min
                    min(np.mean(questions_per_session) / 10, 1.0),  # Normalize to 10 questions
                    np.mean(success_rates),
                    min(len(sessions) / 7, 1.0)  # Normalize to daily usage
                ]
                profile.engagement_score = np.mean(engagement_factors)
                
                # Calculate learning velocity (concepts mastered per week)
                mastered_concepts = sum(1 for mastery in profile.concept_mastery.values() 
                                      if mastery >= self.mastery_threshold)
                weeks_active = len(sessions) / 7
                profile.learning_velocity = mastered_concepts / max(weeks_active, 1)
        
        except Exception as e:
            logger.error(f"❌ Failed to analyze interaction patterns: {e}")
    
    async def calculate_concept_mastery(self, user_id: str, concept: str, 
                                      time_window_days: int = 7) -> ConceptMastery:
        """Calculate detailed concept mastery for a student"""
        try:
            if not self.db_manager:
                return ConceptMastery(concept_name=concept, mastery_level=0.0, confidence_level=0.0)
            
            since_date = datetime.now() - timedelta(days=time_window_days)
            
            async with self.db_manager.postgres.get_connection() as conn:
                # Get interactions related to this concept
                interactions = await conn.fetch("""
                    SELECT success, created_at, execution_time_ms, metadata,
                           request_data, response_data
                    FROM interactions 
                    WHERE user_id = $1 AND agent_type = $2 AND created_at >= $3
                    ORDER BY created_at ASC
                """, user_id, concept, since_date)
            
            if not interactions:
                return ConceptMastery(
                    concept_name=concept,
                    mastery_level=0.0,
                    confidence_level=0.0,
                    attempts_count=0
                )
            
            # Calculate basic metrics
            total_attempts = len(interactions)
            successful_attempts = sum(1 for i in interactions if i['success'])
            success_rate = successful_attempts / total_attempts
            
            # Calculate learning progression
            mastery_progression = []
            window_size = min(5, total_attempts)
            
            for i in range(window_size, total_attempts + 1):
                window_interactions = interactions[i-window_size:i]
                window_success_rate = sum(1 for w in window_interactions if w['success']) / window_size
                mastery_progression.append(window_success_rate)
            
            # Calculate mastery level using weighted recent performance
            if mastery_progression:
                # Weight recent performance more heavily
                weights = np.exp(np.linspace(0, 1, len(mastery_progression)))
                weights = weights / weights.sum()
                mastery_level = np.average(mastery_progression, weights=weights)
            else:
                mastery_level = success_rate
            
            # Calculate confidence based on consistency
            if len(mastery_progression) >= 3:
                confidence_level = 1.0 - np.std(mastery_progression)
            else:
                confidence_level = 0.5 * success_rate  # Lower confidence with few attempts
            
            # Analyze error patterns
            error_patterns = []
            for interaction in interactions:
                if not interaction['success'] and interaction['metadata']:
                    try:
                        metadata = json.loads(interaction['metadata']) if isinstance(interaction['metadata'], str) else interaction['metadata']
                        if 'error_type' in metadata:
                            error_patterns.append(metadata['error_type'])
                    except:
                        pass
            
            # Calculate time to mastery
            time_to_mastery = None
            if mastery_level >= self.mastery_threshold and len(interactions) >= 2:
                first_interaction = interactions[0]['created_at']
                
                # Find when mastery threshold was consistently reached
                for i, progression in enumerate(mastery_progression):
                    if progression >= self.mastery_threshold:
                        mastery_interaction = interactions[i + window_size - 1]
                        time_diff = mastery_interaction['created_at'] - first_interaction
                        time_to_mastery = time_diff.total_seconds() / 3600  # hours
                        break
            
            # Check prerequisites
            prerequisites_mastered = await self._check_prerequisites_mastery(user_id, concept)
            
            return ConceptMastery(
                concept_name=concept,
                mastery_level=max(0.0, min(1.0, mastery_level)),
                confidence_level=max(0.0, min(1.0, confidence_level)),
                time_to_mastery=time_to_mastery,
                attempts_count=total_attempts,
                success_rate=success_rate,
                error_patterns=list(set(error_patterns)),
                prerequisites_mastered=prerequisites_mastered,
                last_interaction=interactions[-1]['created_at']
            )
        
        except Exception as e:
            logger.error(f"❌ Failed to calculate concept mastery for {concept}: {e}")
            return ConceptMastery(concept_name=concept, mastery_level=0.0, confidence_level=0.0)
    
    async def _check_prerequisites_mastery(self, user_id: str, concept: str) -> bool:
        """Check if student has mastered prerequisites for a concept"""
        if not self.concept_graph or concept not in self.concept_graph:
            return True
        
        try:
            # Find prerequisite concepts
            prerequisites = [pred for pred in self.concept_graph.predecessors(concept)
                           if self.concept_graph[pred][concept].get('relationship_type') == 'PREREQUISITE']
            
            if not prerequisites:
                return True
            
            # Check mastery of each prerequisite
            for prereq in prerequisites:
                mastery = await self.calculate_concept_mastery(user_id, prereq, time_window_days=14)
                if mastery.mastery_level < self.mastery_threshold:
                    return False
            
            return True
        
        except Exception as e:
            logger.error(f"❌ Failed to check prerequisites for {concept}: {e}")
            return True
    
    async def track_student_progress(self, user_id: str, time_window_days: int = 30) -> Dict[str, Any]:
        """Comprehensive student progress tracking"""
        try:
            if user_id not in self.student_profiles:
                # Create profile if doesn't exist
                profile = StudentProfile(user_id=user_id)
                await self._load_user_progress(profile)
                self.student_profiles[user_id] = profile
            
            profile = self.student_profiles[user_id]
            
            # Get all concepts the student has interacted with
            since_date = datetime.now() - timedelta(days=time_window_days)
            
            async with self.db_manager.postgres.get_connection() as conn:
                agent_types = await conn.fetch("""
                    SELECT DISTINCT agent_type
                    FROM interactions 
                    WHERE user_id = $1 AND created_at >= $2 AND agent_type IS NOT NULL
                """, user_id, since_date)
            
            concept_masteries = {}
            struggling_concepts = []
            strong_concepts = []
            
            # Calculate mastery for each concept
            for row in agent_types:
                concept = row['agent_type']
                mastery = await self.calculate_concept_mastery(user_id, concept, time_window_days)
                concept_masteries[concept] = mastery
                
                if mastery.mastery_level < self.config['intervention_threshold']:
                    struggling_concepts.append(concept)
                elif mastery.mastery_level >= self.mastery_threshold:
                    strong_concepts.append(concept)
            
            # Update profile
            profile.concept_mastery = {k: v.mastery_level for k, v in concept_masteries.items()}
            profile.struggling_concepts = struggling_concepts
            profile.strong_concepts = strong_concepts
            
            # Calculate overall metrics
            overall_mastery = np.mean(list(profile.concept_mastery.values())) if profile.concept_mastery else 0.0
            
            # Determine current level
            if overall_mastery >= 0.8:
                profile.current_level = "advanced"
            elif overall_mastery >= 0.6:
                profile.current_level = "intermediate"
            else:
                profile.current_level = "beginner"
            
            # Calculate learning trends
            learning_trend = await self._calculate_learning_trend(user_id, time_window_days)
            knowledge_gaps = await self._identify_knowledge_gaps(user_id, concept_masteries)
            
            return {
                'user_id': user_id,
                'overall_mastery': overall_mastery,
                'current_level': profile.current_level,
                'learning_velocity': profile.learning_velocity,
                'engagement_score': profile.engagement_score,
                'concept_masteries': {k: {
                    'mastery_level': v.mastery_level,
                    'confidence_level': v.confidence_level,
                    'attempts_count': v.attempts_count,
                    'success_rate': v.success_rate,
                    'time_to_mastery': v.time_to_mastery,
                    'error_patterns': v.error_patterns
                } for k, v in concept_masteries.items()},
                'struggling_concepts': struggling_concepts,
                'strong_concepts': strong_concepts,
                'learning_trend': learning_trend,
                'knowledge_gaps': knowledge_gaps,
                'session_patterns': profile.session_patterns,
                'last_updated': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"❌ Failed to track student progress for {user_id}: {e}")
            return {'error': str(e)}
    
    async def _calculate_learning_trend(self, user_id: str, time_window_days: int) -> Dict[str, float]:
        """Calculate learning velocity and trend over time"""
        try:
            async with self.db_manager.postgres.get_connection() as conn:
                # Get daily interaction success rates
                daily_stats = await conn.fetch("""
                    SELECT DATE(created_at) as date,
                           COUNT(*) as total_interactions,
                           SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_interactions
                    FROM interactions 
                    WHERE user_id = $1 AND created_at >= $2
                    GROUP BY DATE(created_at)
                    ORDER BY date ASC
                """, user_id, datetime.now() - timedelta(days=time_window_days))
            
            if len(daily_stats) < 3:
                return {'trend': 0.0, 'velocity': 0.0, 'acceleration': 0.0}
            
            # Calculate daily success rates
            dates = [stat['date'] for stat in daily_stats]
            success_rates = [stat['successful_interactions'] / stat['total_interactions'] 
                           for stat in daily_stats]
            
            # Calculate trend using linear regression
            x = np.arange(len(success_rates))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, success_rates)
            
            # Calculate acceleration (second derivative)
            if len(success_rates) >= 3:
                acceleration = np.mean(np.diff(success_rates, 2))
            else:
                acceleration = 0.0
            
            return {
                'trend': slope,  # Positive means improving
                'velocity': np.mean(success_rates[-3:]) if len(success_rates) >= 3 else np.mean(success_rates),
                'acceleration': acceleration,
                'correlation': r_value,
                'significance': p_value
            }
        
        except Exception as e:
            logger.error(f"❌ Failed to calculate learning trend: {e}")
            return {'trend': 0.0, 'velocity': 0.0, 'acceleration': 0.0}
    
    async def _identify_knowledge_gaps(self, user_id: str, concept_masteries: Dict[str, ConceptMastery]) -> List[Dict[str, Any]]:
        """Identify knowledge gaps and prerequisite deficiencies"""
        gaps = []
        
        if not self.concept_graph:
            return gaps
        
        try:
            for concept, mastery in concept_masteries.items():
                if concept not in self.concept_graph:
                    continue
                
                # If student is struggling with this concept
                if mastery.mastery_level < self.config['intervention_threshold']:
                    # Check if prerequisites are the issue
                    prerequisites = [pred for pred in self.concept_graph.predecessors(concept)
                                   if self.concept_graph[pred][concept].get('relationship_type') == 'PREREQUISITE']
                    
                    prerequisite_gaps = []
                    for prereq in prerequisites:
                        if prereq in concept_masteries:
                            prereq_mastery = concept_masteries[prereq]
                            if prereq_mastery.mastery_level < self.mastery_threshold:
                                prerequisite_gaps.append({
                                    'concept': prereq,
                                    'mastery_level': prereq_mastery.mastery_level,
                                    'gap_severity': self.mastery_threshold - prereq_mastery.mastery_level
                                })
                    
                    if prerequisite_gaps or mastery.mastery_level < 0.3:
                        gaps.append({
                            'concept': concept,
                            'mastery_level': mastery.mastery_level,
                            'confidence_level': mastery.confidence_level,
                            'gap_type': 'prerequisite' if prerequisite_gaps else 'conceptual',
                            'prerequisite_gaps': prerequisite_gaps,
                            'recommended_action': self._recommend_intervention(mastery, prerequisite_gaps)
                        })
            
            # Sort by severity (lowest mastery first)
            gaps.sort(key=lambda x: x['mastery_level'])
            
            return gaps
        
        except Exception as e:
            logger.error(f"❌ Failed to identify knowledge gaps: {e}")
            return []
    
    def _recommend_intervention(self, mastery: ConceptMastery, prerequisite_gaps: List[Dict]) -> str:
        """Recommend intervention strategy based on mastery analysis"""
        if prerequisite_gaps:
            return f"Review prerequisites: {', '.join([gap['concept'] for gap in prerequisite_gaps])}"
        elif mastery.confidence_level < 0.5:
            return "Practice with varied examples to build confidence"
        elif mastery.attempts_count < self.config['min_interactions_for_mastery']:
            return "Need more practice - increase interaction frequency"
        elif mastery.error_patterns:
            most_common_error = Counter(mastery.error_patterns).most_common(1)[0][0]
            return f"Focus on addressing {most_common_error} errors"
        else:
            return "Structured review and guided practice recommended"
    
    async def detect_learning_difficulties(self, user_id: str) -> Dict[str, Any]:
        """Detect early warning signs for learning difficulties"""
        try:
            # Analyze recent performance patterns
            since_date = datetime.now() - timedelta(days=14)
            
            async with self.db_manager.postgres.get_connection() as conn:
                interactions = await conn.fetch("""
                    SELECT agent_type, success, created_at, execution_time_ms
                    FROM interactions 
                    WHERE user_id = $1 AND created_at >= $2
                    ORDER BY created_at ASC
                """, user_id, since_date)
            
            if len(interactions) < 10:
                return {'warning_level': 'insufficient_data', 'recommendations': []}
            
            # Calculate warning indicators
            warnings = []
            warning_level = 'none'
            
            # 1. Declining success rate
            recent_success = [i['success'] for i in interactions[-10:]]
            earlier_success = [i['success'] for i in interactions[-20:-10]] if len(interactions) >= 20 else []
            
            if earlier_success and np.mean(recent_success) < np.mean(earlier_success) - 0.2:
                warnings.append('declining_performance')
                warning_level = 'moderate'
            
            # 2. Excessive time per interaction
            response_times = [i['execution_time_ms'] for i in interactions if i['execution_time_ms']]
            if response_times:
                avg_time = np.mean(response_times)
                if avg_time > 30000:  # 30 seconds average
                    warnings.append('slow_response_pattern')
                    warning_level = 'moderate' if warning_level == 'none' else 'high'
            
            # 3. Concept switching without mastery
            concept_sequence = [i['agent_type'] for i in interactions]
            concept_switches = sum(1 for i in range(1, len(concept_sequence)) 
                                 if concept_sequence[i] != concept_sequence[i-1])
            
            if concept_switches > len(set(concept_sequence)) * 3:
                warnings.append('excessive_concept_switching')
                warning_level = 'moderate' if warning_level == 'none' else 'high'
            
            # 4. Low engagement pattern
            daily_interactions = defaultdict(int)
            for interaction in interactions:
                day = interaction['created_at'].date()
                daily_interactions[day] += 1
            
            avg_daily_interactions = np.mean(list(daily_interactions.values())) if daily_interactions else 0
            if avg_daily_interactions < 2:
                warnings.append('low_engagement')
            
            # Generate recommendations
            recommendations = self._generate_intervention_recommendations(warnings, user_id)
            
            return {
                'user_id': user_id,
                'warning_level': warning_level,
                'warning_indicators': warnings,
                'recommendations': recommendations,
                'assessment_date': datetime.now().isoformat(),
                'data_period_days': 14
            }
        
        except Exception as e:
            logger.error(f"❌ Failed to detect learning difficulties for {user_id}: {e}")
            return {'warning_level': 'error', 'error': str(e)}
    
    def _generate_intervention_recommendations(self, warnings: List[str], user_id: str) -> List[str]:
        """Generate specific intervention recommendations based on warning indicators"""
        recommendations = []
        
        if 'declining_performance' in warnings:
            recommendations.extend([
                "Schedule a review session of recent topics",
                "Consider reducing difficulty level temporarily",
                "Provide additional scaffolding and hints"
            ])
        
        if 'slow_response_pattern' in warnings:
            recommendations.extend([
                "Break down complex problems into smaller steps",
                "Provide more worked examples",
                "Consider one-on-one tutoring support"
            ])
        
        if 'excessive_concept_switching' in warnings:
            recommendations.extend([
                "Focus on mastering one concept before moving to next",
                "Provide clearer learning objectives",
                "Implement prerequisite checking"
            ])
        
        if 'low_engagement' in warnings:
            recommendations.extend([
                "Introduce gamification elements",
                "Provide more varied problem types",
                "Check for external factors affecting participation"
            ])
        
        return recommendations

    async def calculate_learning_efficiency(self, user_id: str) -> Dict[str, float]:
        """Calculate learning efficiency metrics for a student"""
        try:
            async with self.db_manager.postgres.get_connection() as conn:
                # Get all interactions for analysis
                interactions = await conn.fetch("""
                    SELECT agent_type, success, created_at, execution_time_ms
                    FROM interactions 
                    WHERE user_id = $1 AND created_at >= $2
                    ORDER BY created_at ASC
                """, user_id, datetime.now() - timedelta(days=30))
            
            if not interactions:
                return {'efficiency_score': 0.0, 'time_efficiency': 0.0, 'success_efficiency': 0.0}
            
            # Calculate success efficiency (success rate improvement over time)
            success_sequence = [i['success'] for i in interactions]
            window_size = min(10, len(success_sequence) // 2)
            
            if len(success_sequence) >= window_size * 2:
                early_success_rate = np.mean(success_sequence[:window_size])
                recent_success_rate = np.mean(success_sequence[-window_size:])
                success_efficiency = max(0.0, recent_success_rate - early_success_rate + early_success_rate)
            else:
                success_efficiency = np.mean(success_sequence)
            
            # Calculate time efficiency (decreasing time to completion)
            response_times = [i['execution_time_ms'] for i in interactions if i['execution_time_ms']]
            if len(response_times) >= 10:
                early_avg_time = np.mean(response_times[:len(response_times)//2])
                recent_avg_time = np.mean(response_times[len(response_times)//2:])
                time_efficiency = max(0.0, min(1.0, early_avg_time / max(recent_avg_time, 1000)))
            else:
                time_efficiency = 0.5  # Neutral score
            
            # Overall efficiency score
            efficiency_score = (success_efficiency * 0.7 + time_efficiency * 0.3)
            
            return {
                'efficiency_score': efficiency_score,
                'success_efficiency': success_efficiency,
                'time_efficiency': time_efficiency,
                'total_interactions': len(interactions),
                'calculation_date': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"❌ Failed to calculate learning efficiency for {user_id}: {e}")
            return {'efficiency_score': 0.0, 'error': str(e)}

# Example usage and testing
async def test_learning_analytics():
    """Test function for learning analytics engine"""
    try:
        logger.info("🧪 Testing Learning Analytics Engine")
        
        # Initialize without database for basic testing
        engine = LearningAnalyticsEngine()
        await engine.initialize()
        
        # Test concept mastery calculation
        sample_mastery = ConceptMastery(
            concept_name="kinematics",
            mastery_level=0.75,
            confidence_level=0.80,
            attempts_count=15,
            success_rate=0.75,
            time_to_mastery=2.5
        )
        
        logger.info(f"✅ Sample concept mastery: {sample_mastery}")
        
        # Test student profile
        sample_profile = StudentProfile(
            user_id="test_user",
            current_level="intermediate",
            learning_velocity=0.5,
            engagement_score=0.8
        )
        
        logger.info(f"✅ Sample student profile: {sample_profile}")
        
        logger.info("✅ Learning Analytics Engine test completed")
        
    except Exception as e:
        logger.error(f"❌ Learning Analytics test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_learning_analytics())