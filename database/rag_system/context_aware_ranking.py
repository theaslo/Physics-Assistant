#!/usr/bin/env python3
"""
Context-Aware Ranking and Filtering System for Educational Content
Provides personalized content ranking based on student profiles, learning history, and educational context
"""
import os
import json
import logging
import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import math
from collections import defaultdict, Counter

# Third-party imports
import redis
from neo4j import GraphDatabase
import asyncpg

# Local imports
from .semantic_search import SearchResult, SearchQuery

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LearningStyle(Enum):
    """Different learning style preferences"""
    VISUAL = "visual"
    AUDITORY = "auditory"
    KINESTHETIC = "kinesthetic"
    READ_WRITE = "read_write"
    MULTIMODAL = "multimodal"

class ContentPreference(Enum):
    """Content type preferences"""
    CONCEPTUAL = "conceptual"
    PRACTICAL = "practical"
    THEORETICAL = "theoretical"
    PROBLEM_SOLVING = "problem_solving"
    VISUAL_DIAGRAMS = "visual_diagrams"
    MATHEMATICAL = "mathematical"

@dataclass
class StudentProfile:
    """Comprehensive student learning profile"""
    user_id: str
    current_level: str = "intermediate"  # beginner, intermediate, advanced
    learning_style: LearningStyle = LearningStyle.MULTIMODAL
    content_preferences: List[ContentPreference] = field(default_factory=list)
    
    # Performance metrics
    avg_response_time: float = 0.0  # seconds
    success_rate: float = 0.0  # 0-1
    engagement_score: float = 0.0  # 0-1
    
    # Learning patterns
    preferred_difficulty_progression: str = "gradual"  # gradual, steep, mixed
    attention_span: int = 30  # minutes
    preferred_session_length: int = 45  # minutes
    
    # Topic-specific performance
    topic_mastery: Dict[str, float] = field(default_factory=dict)  # topic -> mastery score
    topic_interest: Dict[str, float] = field(default_factory=dict)  # topic -> interest score
    concept_understanding: Dict[str, float] = field(default_factory=dict)  # concept -> understanding
    
    # Learning history
    completed_topics: List[str] = field(default_factory=list)
    struggling_concepts: List[str] = field(default_factory=list)
    recent_queries: List[str] = field(default_factory=list)
    learning_goals: List[str] = field(default_factory=list)
    
    # Temporal patterns
    active_hours: List[int] = field(default_factory=lambda: list(range(9, 17)))  # 9 AM - 5 PM
    productive_days: List[str] = field(default_factory=lambda: ["monday", "tuesday", "wednesday", "thursday", "friday"])
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass 
class LearningContext:
    """Current learning session context"""
    session_id: str
    user_id: str
    current_topic: str = ""
    learning_objective: str = ""
    session_duration: int = 0  # minutes elapsed
    previous_queries: List[str] = field(default_factory=list)
    current_difficulty_level: str = "intermediate"
    
    # Session state
    concepts_covered: List[str] = field(default_factory=list)
    problems_attempted: int = 0
    problems_solved: int = 0
    help_requests: int = 0
    
    # Context clues
    query_complexity_trend: str = "stable"  # increasing, decreasing, stable
    error_patterns: List[str] = field(default_factory=list)
    confusion_indicators: List[str] = field(default_factory=list)
    
    # Environmental context
    time_of_day: int = field(default_factory=lambda: datetime.now().hour)
    day_of_week: str = field(default_factory=lambda: datetime.now().strftime("%A").lower())
    estimated_focus_level: float = 1.0  # 0-1, based on session patterns

@dataclass
class ContentMetrics:
    """Metrics for educational content evaluation"""
    educational_value: float = 0.0
    difficulty_score: float = 0.0
    engagement_potential: float = 0.0
    prerequisite_coverage: float = 0.0
    concept_clarity: float = 0.0
    practical_relevance: float = 0.0
    
    # Content-specific metrics
    formula_complexity: float = 0.0
    diagram_quality: float = 0.0
    example_quality: float = 0.0
    explanation_depth: float = 0.0

class StudentProfileManager:
    """Manages student learning profiles and adaptive updates"""
    
    def __init__(self, postgres_pool: asyncpg.Pool, redis_client: redis.Redis):
        self.postgres_pool = postgres_pool
        self.redis_client = redis_client
        self.profile_cache = {}
        self.cache_expiry = 3600  # 1 hour
    
    async def get_student_profile(self, user_id: str) -> Optional[StudentProfile]:
        """Get or create student profile"""
        
        # Check cache first
        cached_profile = await self._get_cached_profile(user_id)
        if cached_profile:
            return cached_profile
        
        # Load from database
        profile = await self._load_profile_from_db(user_id)
        
        if not profile:
            # Create new profile
            profile = await self._create_default_profile(user_id)
        
        # Cache profile
        await self._cache_profile(profile)
        
        return profile
    
    async def update_student_profile(self, user_id: str, interaction_data: Dict[str, Any]):
        """Update student profile based on new interaction"""
        
        profile = await self.get_student_profile(user_id)
        if not profile:
            return
        
        # Update profile based on interaction
        await self._update_performance_metrics(profile, interaction_data)
        await self._update_topic_mastery(profile, interaction_data)
        await self._update_learning_patterns(profile, interaction_data)
        await self._update_preferences(profile, interaction_data)
        
        # Save updated profile
        await self._save_profile_to_db(profile)
        await self._cache_profile(profile)
    
    async def _get_cached_profile(self, user_id: str) -> Optional[StudentProfile]:
        """Get profile from cache"""
        try:
            cache_key = f"student_profile:{user_id}"
            cached_data = await asyncio.to_thread(self.redis_client.get, cache_key)
            
            if cached_data:
                profile_dict = json.loads(cached_data)
                return self._dict_to_profile(profile_dict)
                
        except Exception as e:
            logger.warning(f"Failed to get cached profile: {e}")
        
        return None
    
    async def _cache_profile(self, profile: StudentProfile):
        """Cache profile in Redis"""
        try:
            cache_key = f"student_profile:{profile.user_id}"
            profile_dict = self._profile_to_dict(profile)
            profile_json = json.dumps(profile_dict, default=str)
            
            await asyncio.to_thread(
                self.redis_client.setex,
                cache_key,
                self.cache_expiry,
                profile_json
            )
            
        except Exception as e:
            logger.warning(f"Failed to cache profile: {e}")
    
    async def _load_profile_from_db(self, user_id: str) -> Optional[StudentProfile]:
        """Load profile from PostgreSQL"""
        
        query = """
        SELECT profile_data, learning_metrics, preferences, last_updated
        FROM student_profiles
        WHERE user_id = $1
        """
        
        try:
            async with self.postgres_pool.acquire() as conn:
                row = await conn.fetchrow(query, user_id)
                
                if row:
                    profile_data = row['profile_data']
                    profile = self._dict_to_profile(profile_data)
                    profile.last_updated = row['last_updated']
                    return profile
                    
        except Exception as e:
            logger.error(f"Failed to load profile from DB: {e}")
        
        return None
    
    async def _save_profile_to_db(self, profile: StudentProfile):
        """Save profile to PostgreSQL"""
        
        profile_dict = self._profile_to_dict(profile)
        
        query = """
        INSERT INTO student_profiles (user_id, profile_data, learning_metrics, preferences, last_updated)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (user_id) DO UPDATE SET
            profile_data = EXCLUDED.profile_data,
            learning_metrics = EXCLUDED.learning_metrics,
            preferences = EXCLUDED.preferences,
            last_updated = EXCLUDED.last_updated
        """
        
        try:
            async with self.postgres_pool.acquire() as conn:
                await conn.execute(
                    query,
                    profile.user_id,
                    json.dumps(profile_dict),
                    json.dumps({
                        'success_rate': profile.success_rate,
                        'engagement_score': profile.engagement_score,
                        'avg_response_time': profile.avg_response_time
                    }),
                    json.dumps({
                        'learning_style': profile.learning_style.value,
                        'content_preferences': [p.value for p in profile.content_preferences]
                    }),
                    datetime.now()
                )
                
        except Exception as e:
            logger.error(f"Failed to save profile to DB: {e}")
    
    async def _create_default_profile(self, user_id: str) -> StudentProfile:
        """Create default profile for new user"""
        
        # Get initial assessment data if available
        initial_data = await self._get_initial_assessment_data(user_id)
        
        profile = StudentProfile(
            user_id=user_id,
            current_level=initial_data.get('level', 'intermediate'),
            learning_style=LearningStyle(initial_data.get('learning_style', 'multimodal')),
            content_preferences=[ContentPreference.CONCEPTUAL, ContentPreference.PRACTICAL],
            avg_response_time=30.0,
            success_rate=0.5,
            engagement_score=0.7,
            preferred_difficulty_progression="gradual"
        )
        
        return profile
    
    async def _get_initial_assessment_data(self, user_id: str) -> Dict[str, Any]:
        """Get any initial assessment or onboarding data"""
        
        try:
            query = """
            SELECT assessment_data
            FROM user_assessments
            WHERE user_id = $1
            ORDER BY created_at DESC
            LIMIT 1
            """
            
            async with self.postgres_pool.acquire() as conn:
                row = await conn.fetchrow(query, user_id)
                
                if row:
                    return row['assessment_data']
                    
        except Exception as e:
            logger.warning(f"No initial assessment data found: {e}")
        
        return {}
    
    async def _update_performance_metrics(self, profile: StudentProfile, interaction_data: Dict):
        """Update performance metrics based on interaction"""
        
        # Update success rate
        if 'success' in interaction_data:
            current_success = profile.success_rate
            new_success = 1.0 if interaction_data['success'] else 0.0
            profile.success_rate = (current_success * 0.9) + (new_success * 0.1)
        
        # Update response time
        if 'response_time' in interaction_data:
            current_time = profile.avg_response_time
            new_time = interaction_data['response_time']
            profile.avg_response_time = (current_time * 0.8) + (new_time * 0.2)
        
        # Update engagement score based on session length and interactions
        if 'session_duration' in interaction_data:
            duration = interaction_data['session_duration']
            # Higher engagement for longer, productive sessions
            engagement_boost = min(0.1, duration / 600)  # Max 0.1 for 10+ minutes
            profile.engagement_score = min(1.0, profile.engagement_score + engagement_boost * 0.1)
    
    async def _update_topic_mastery(self, profile: StudentProfile, interaction_data: Dict):
        """Update topic mastery scores"""
        
        topic = interaction_data.get('topic')
        if not topic:
            return
        
        # Update mastery based on success rate in this topic
        success = interaction_data.get('success', False)
        current_mastery = profile.topic_mastery.get(topic, 0.5)
        
        if success:
            new_mastery = min(1.0, current_mastery + 0.05)
        else:
            new_mastery = max(0.0, current_mastery - 0.02)
        
        profile.topic_mastery[topic] = new_mastery
        
        # Update concept understanding
        concepts = interaction_data.get('concepts', [])
        for concept in concepts:
            current_understanding = profile.concept_understanding.get(concept, 0.5)
            
            if success:
                profile.concept_understanding[concept] = min(1.0, current_understanding + 0.03)
            else:
                profile.concept_understanding[concept] = max(0.0, current_understanding - 0.01)
                
                # Add to struggling concepts if understanding is low
                if profile.concept_understanding[concept] < 0.3:
                    if concept not in profile.struggling_concepts:
                        profile.struggling_concepts.append(concept)
    
    async def _update_learning_patterns(self, profile: StudentProfile, interaction_data: Dict):
        """Update learning pattern preferences"""
        
        # Update preferred session length based on productive sessions
        if interaction_data.get('productive_session'):
            session_length = interaction_data.get('session_duration', 30)
            current_preference = profile.preferred_session_length
            profile.preferred_session_length = int((current_preference * 0.9) + (session_length * 0.1))
        
        # Update difficulty progression preference
        difficulty_changes = interaction_data.get('difficulty_changes', [])
        if difficulty_changes:
            # Analyze if gradual or steep changes work better
            success_with_gradual = sum(1 for change in difficulty_changes if change['type'] == 'gradual' and change['success'])
            success_with_steep = sum(1 for change in difficulty_changes if change['type'] == 'steep' and change['success'])
            
            if success_with_gradual > success_with_steep:
                profile.preferred_difficulty_progression = "gradual"
            elif success_with_steep > success_with_gradual:
                profile.preferred_difficulty_progression = "steep"
    
    async def _update_preferences(self, profile: StudentProfile, interaction_data: Dict):
        """Update content preferences based on engagement"""
        
        content_type = interaction_data.get('content_type')
        engagement = interaction_data.get('engagement_score', 0.5)
        
        if content_type and engagement > 0.7:  # High engagement threshold
            try:
                preference = ContentPreference(content_type.lower())
                if preference not in profile.content_preferences:
                    profile.content_preferences.append(preference)
            except ValueError:
                pass  # Invalid content type
        
        # Update recent queries
        query = interaction_data.get('query')
        if query:
            profile.recent_queries.append(query)
            # Keep only last 10 queries
            if len(profile.recent_queries) > 10:
                profile.recent_queries = profile.recent_queries[-10:]
    
    def _profile_to_dict(self, profile: StudentProfile) -> Dict:
        """Convert profile to dictionary for serialization"""
        return {
            'user_id': profile.user_id,
            'current_level': profile.current_level,
            'learning_style': profile.learning_style.value,
            'content_preferences': [p.value for p in profile.content_preferences],
            'avg_response_time': profile.avg_response_time,
            'success_rate': profile.success_rate,
            'engagement_score': profile.engagement_score,
            'preferred_difficulty_progression': profile.preferred_difficulty_progression,
            'attention_span': profile.attention_span,
            'preferred_session_length': profile.preferred_session_length,
            'topic_mastery': profile.topic_mastery,
            'topic_interest': profile.topic_interest,
            'concept_understanding': profile.concept_understanding,
            'completed_topics': profile.completed_topics,
            'struggling_concepts': profile.struggling_concepts,
            'recent_queries': profile.recent_queries,
            'learning_goals': profile.learning_goals,
            'active_hours': profile.active_hours,
            'productive_days': profile.productive_days,
            'created_at': profile.created_at.isoformat(),
            'last_updated': profile.last_updated.isoformat()
        }
    
    def _dict_to_profile(self, data: Dict) -> StudentProfile:
        """Convert dictionary to profile object"""
        return StudentProfile(
            user_id=data['user_id'],
            current_level=data.get('current_level', 'intermediate'),
            learning_style=LearningStyle(data.get('learning_style', 'multimodal')),
            content_preferences=[ContentPreference(p) for p in data.get('content_preferences', [])],
            avg_response_time=data.get('avg_response_time', 30.0),
            success_rate=data.get('success_rate', 0.5),
            engagement_score=data.get('engagement_score', 0.7),
            preferred_difficulty_progression=data.get('preferred_difficulty_progression', 'gradual'),
            attention_span=data.get('attention_span', 30),
            preferred_session_length=data.get('preferred_session_length', 45),
            topic_mastery=data.get('topic_mastery', {}),
            topic_interest=data.get('topic_interest', {}),
            concept_understanding=data.get('concept_understanding', {}),
            completed_topics=data.get('completed_topics', []),
            struggling_concepts=data.get('struggling_concepts', []),
            recent_queries=data.get('recent_queries', []),
            learning_goals=data.get('learning_goals', []),
            active_hours=data.get('active_hours', list(range(9, 17))),
            productive_days=data.get('productive_days', ["monday", "tuesday", "wednesday", "thursday", "friday"]),
            created_at=datetime.fromisoformat(data.get('created_at', datetime.now().isoformat())),
            last_updated=datetime.fromisoformat(data.get('last_updated', datetime.now().isoformat()))
        )

class ContentAnalyzer:
    """Analyzes content to extract educational metrics"""
    
    def __init__(self, neo4j_driver):
        self.neo4j_driver = neo4j_driver
        
        # Physics concept difficulty mapping
        self.concept_difficulty = {
            # Basic mechanics
            'position': 1, 'velocity': 2, 'acceleration': 3,
            'force': 2, 'newton_laws': 3, 'friction': 3,
            'work': 3, 'energy': 4, 'power': 3,
            'momentum': 4, 'collision': 5, 'impulse': 4,
            
            # Advanced mechanics
            'rotation': 5, 'torque': 5, 'angular_momentum': 6,
            'oscillation': 6, 'wave': 6, 'resonance': 7,
            
            # Thermodynamics
            'temperature': 2, 'heat': 3, 'entropy': 7, 'thermal_equilibrium': 5,
            
            # Electromagnetism
            'electric_field': 4, 'magnetic_field': 5, 'electromagnetic_induction': 7
        }
    
    async def analyze_content(self, content: SearchResult) -> ContentMetrics:
        """Analyze content and generate educational metrics"""
        
        metrics = ContentMetrics()
        
        # Basic metrics from content type
        if content.content_type == 'explanation':
            metrics.educational_value = 0.8
            metrics.concept_clarity = 0.7
            metrics.explanation_depth = 0.8
        elif content.content_type == 'problem':
            metrics.educational_value = 0.9
            metrics.practical_relevance = 0.9
            metrics.example_quality = 0.7
        elif content.content_type == 'formula':
            metrics.educational_value = 0.6
            metrics.formula_complexity = await self._analyze_formula_complexity(content.content)
        elif content.content_type == 'concept':
            metrics.educational_value = 0.7
            metrics.concept_clarity = 0.8
        
        # Analyze difficulty
        metrics.difficulty_score = await self._analyze_difficulty(content)
        
        # Analyze engagement potential
        metrics.engagement_potential = await self._analyze_engagement_potential(content)
        
        # Analyze prerequisite coverage
        metrics.prerequisite_coverage = await self._analyze_prerequisite_coverage(content.node_id)
        
        return metrics
    
    async def _analyze_difficulty(self, content: SearchResult) -> float:
        """Analyze content difficulty"""
        
        # Get difficulty from metadata if available
        if 'difficulty_level' in content.metadata:
            difficulty_map = {'beginner': 0.2, 'intermediate': 0.5, 'advanced': 0.8}
            return difficulty_map.get(content.metadata['difficulty_level'], 0.5)
        
        # Analyze based on content
        difficulty_indicators = [
            'derivative', 'integral', 'calculus', 'differential',
            'complex', 'advanced', 'quantum', 'relativistic',
            'theoretical', 'abstract', 'proof', 'derive'
        ]
        
        content_lower = content.content.lower()
        difficulty_count = sum(1 for indicator in difficulty_indicators if indicator in content_lower)
        
        # Normalize to 0-1 scale
        return min(0.9, difficulty_count * 0.15)
    
    async def _analyze_engagement_potential(self, content: SearchResult) -> float:
        """Analyze how engaging the content is likely to be"""
        
        engagement_score = 0.5  # Base score
        
        content_lower = content.content.lower()
        
        # Positive engagement indicators
        positive_indicators = [
            'example', 'real-world', 'application', 'everyday',
            'experiment', 'demonstration', 'interactive', 'visual',
            'diagram', 'graph', 'animation', 'simulation',
            'practical', 'hands-on', 'try', 'explore'
        ]
        
        # Negative engagement indicators
        negative_indicators = [
            'memorize', 'abstract', 'theoretical', 'complex',
            'difficult', 'advanced', 'complicated'
        ]
        
        positive_count = sum(1 for indicator in positive_indicators if indicator in content_lower)
        negative_count = sum(1 for indicator in negative_indicators if indicator in content_lower)
        
        engagement_score += (positive_count * 0.1) - (negative_count * 0.05)
        
        # Content type adjustments
        if content.content_type == 'problem':
            engagement_score += 0.2  # Problems are generally more engaging
        elif content.content_type == 'explanation' and 'visual' in content_lower:
            engagement_score += 0.15  # Visual explanations are engaging
        
        return max(0.1, min(1.0, engagement_score))
    
    async def _analyze_formula_complexity(self, formula_content: str) -> float:
        """Analyze mathematical complexity of formulas"""
        
        if not formula_content:
            return 0.0
        
        complexity_score = 0.0
        
        # Count mathematical operations
        operations = ['+', '-', '*', '/', '^', '√', '∫', '∂', 'sin', 'cos', 'tan', 'log', 'ln']
        operation_count = sum(formula_content.count(op) for op in operations)
        complexity_score += operation_count * 0.05
        
        # Count variables
        import re
        variables = re.findall(r'[a-zA-Z]+', formula_content)
        unique_variables = len(set(variables))
        complexity_score += unique_variables * 0.03
        
        # Special complexity indicators
        if '∫' in formula_content or 'integral' in formula_content.lower():
            complexity_score += 0.3
        if '∂' in formula_content or 'partial' in formula_content.lower():
            complexity_score += 0.25
        if any(trig in formula_content.lower() for trig in ['sin', 'cos', 'tan']):
            complexity_score += 0.1
        
        return min(1.0, complexity_score)
    
    async def _analyze_prerequisite_coverage(self, node_id: int) -> float:
        """Analyze how well prerequisites are covered"""
        
        query = """
        MATCH (n)<-[:PREREQUISITE_FOR]-(prereq)
        WHERE id(n) = $node_id
        RETURN count(prereq) as prerequisite_count,
               collect(prereq.name) as prerequisites
        """
        
        def run_query():
            with self.neo4j_driver.session() as session:
                result = session.run(query, {'node_id': node_id})
                record = result.single()
                return dict(record) if record else {'prerequisite_count': 0, 'prerequisites': []}
        
        try:
            result = await asyncio.to_thread(run_query)
            
            prerequisite_count = result['prerequisite_count']
            
            # More prerequisites generally mean better coverage
            # But too many might be overwhelming
            if prerequisite_count == 0:
                return 0.3  # Standalone content
            elif 1 <= prerequisite_count <= 3:
                return 0.8  # Good prerequisite coverage
            elif 4 <= prerequisite_count <= 6:
                return 0.6  # Moderate coverage
            else:
                return 0.4  # Might be too complex
                
        except Exception as e:
            logger.error(f"Failed to analyze prerequisite coverage: {e}")
            return 0.5

class ContextAwareRanker:
    """Main ranking system that combines all factors"""
    
    def __init__(self, profile_manager: StudentProfileManager, content_analyzer: ContentAnalyzer):
        self.profile_manager = profile_manager
        self.content_analyzer = content_analyzer
    
    async def rank_results(self, results: List[SearchResult], user_id: str, 
                          context: LearningContext) -> List[SearchResult]:
        """Rank search results based on student profile and context"""
        
        if not results:
            return []
        
        # Get student profile
        profile = await self.profile_manager.get_student_profile(user_id)
        if not profile:
            logger.warning(f"No profile found for user {user_id}")
            return results  # Return unmodified if no profile
        
        # Analyze content and calculate personalized scores
        enhanced_results = []
        
        for result in results:
            # Analyze content metrics
            content_metrics = await self.content_analyzer.analyze_content(result)
            
            # Calculate personalized score
            personalized_score = await self._calculate_personalized_score(
                result, content_metrics, profile, context
            )
            
            # Create enhanced result
            result.similarity_score = personalized_score
            result.metadata['content_metrics'] = content_metrics
            result.metadata['personalization_factors'] = await self._get_personalization_explanation(
                result, profile, context
            )
            
            enhanced_results.append(result)
        
        # Sort by personalized score
        enhanced_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(enhanced_results):
            result.rank = i + 1
        
        return enhanced_results
    
    async def _calculate_personalized_score(self, result: SearchResult, 
                                          content_metrics: ContentMetrics,
                                          profile: StudentProfile, 
                                          context: LearningContext) -> float:
        """Calculate personalized relevance score"""
        
        # Base semantic similarity score
        base_score = result.similarity_score
        
        # Difficulty alignment score
        difficulty_score = self._calculate_difficulty_alignment(
            content_metrics.difficulty_score, profile, context
        )
        
        # Content preference score
        preference_score = self._calculate_content_preference_score(
            result, profile
        )
        
        # Learning path score
        learning_path_score = await self._calculate_learning_path_score(
            result, profile, context
        )
        
        # Temporal relevance score
        temporal_score = self._calculate_temporal_relevance(
            result, profile, context
        )
        
        # Mastery-based score
        mastery_score = self._calculate_mastery_based_score(
            result, profile
        )
        
        # Engagement prediction score
        engagement_score = self._calculate_engagement_prediction(
            content_metrics, profile
        )
        
        # Combine scores with weights
        weights = {
            'base': 0.25,
            'difficulty': 0.20,
            'preference': 0.15,
            'learning_path': 0.15,
            'temporal': 0.10,
            'mastery': 0.10,
            'engagement': 0.05
        }
        
        total_score = (
            base_score * weights['base'] +
            difficulty_score * weights['difficulty'] +
            preference_score * weights['preference'] +
            learning_path_score * weights['learning_path'] +
            temporal_score * weights['temporal'] +
            mastery_score * weights['mastery'] +
            engagement_score * weights['engagement']
        )
        
        return max(0.0, min(1.0, total_score))
    
    def _calculate_difficulty_alignment(self, content_difficulty: float, 
                                      profile: StudentProfile, 
                                      context: LearningContext) -> float:
        """Calculate how well content difficulty matches student level"""
        
        # Map student levels to difficulty preferences
        level_preferences = {
            'beginner': 0.2,
            'intermediate': 0.5,
            'advanced': 0.8
        }
        
        preferred_difficulty = level_preferences.get(profile.current_level, 0.5)
        
        # Consider context adjustments
        if context.query_complexity_trend == "increasing":
            preferred_difficulty += 0.1  # Student might be ready for harder content
        elif context.query_complexity_trend == "decreasing":
            preferred_difficulty -= 0.1  # Student might need easier content
        
        # Consider recent performance
        if profile.success_rate > 0.8:
            preferred_difficulty += 0.05  # High performer, can handle slightly harder
        elif profile.success_rate < 0.4:
            preferred_difficulty -= 0.05  # Struggling, need easier content
        
        # Calculate alignment score (higher score for closer match)
        difficulty_diff = abs(content_difficulty - preferred_difficulty)
        alignment_score = 1.0 - (difficulty_diff * 2)  # Scale difference
        
        return max(0.0, alignment_score)
    
    def _calculate_content_preference_score(self, result: SearchResult, 
                                          profile: StudentProfile) -> float:
        """Calculate score based on content type preferences"""
        
        if not profile.content_preferences:
            return 0.5  # Neutral if no preferences
        
        content_type_mapping = {
            'concept': ContentPreference.CONCEPTUAL,
            'problem': ContentPreference.PROBLEM_SOLVING,
            'formula': ContentPreference.MATHEMATICAL,
            'explanation': ContentPreference.THEORETICAL
        }
        
        content_preference = content_type_mapping.get(result.content_type)
        
        if content_preference in profile.content_preferences:
            return 0.8
        
        # Check for visual indicators if visual preference
        if ContentPreference.VISUAL_DIAGRAMS in profile.content_preferences:
            visual_indicators = ['diagram', 'graph', 'visual', 'figure', 'image']
            if any(indicator in result.content.lower() for indicator in visual_indicators):
                return 0.7
        
        return 0.3  # Lower score if not preferred
    
    async def _calculate_learning_path_score(self, result: SearchResult,
                                           profile: StudentProfile,
                                           context: LearningContext) -> float:
        """Calculate score based on learning path and prerequisites"""
        
        score = 0.5  # Base score
        
        # Check if content relates to current topic
        if context.current_topic:
            content_lower = result.content.lower()
            topic_lower = context.current_topic.lower()
            
            if topic_lower in content_lower or any(
                topic_word in content_lower 
                for topic_word in topic_lower.split()
            ):
                score += 0.3
        
        # Check if content addresses struggling concepts
        for struggling_concept in profile.struggling_concepts:
            if struggling_concept.lower() in result.content.lower():
                score += 0.2
                break  # Only boost once
        
        # Check if content builds on mastered topics
        mastered_topics = [topic for topic, mastery in profile.topic_mastery.items() 
                          if mastery > 0.7]
        
        for mastered_topic in mastered_topics:
            if mastered_topic.lower() in result.content.lower():
                score += 0.1
                break
        
        return min(1.0, score)
    
    def _calculate_temporal_relevance(self, result: SearchResult,
                                    profile: StudentProfile,
                                    context: LearningContext) -> float:
        """Calculate temporal relevance based on time and session context"""
        
        score = 0.5  # Base score
        
        # Time of day preferences
        current_hour = context.time_of_day
        if current_hour in profile.active_hours:
            score += 0.2
        
        # Day of week preferences
        current_day = context.day_of_week
        if current_day in profile.productive_days:
            score += 0.1
        
        # Session duration considerations
        if context.session_duration > profile.attention_span:
            # Student might be getting tired, prefer easier content
            if 'difficulty_level' in result.metadata:
                if result.metadata['difficulty_level'] == 'beginner':
                    score += 0.2
        
        # Focus level adjustments
        score *= context.estimated_focus_level
        
        return max(0.1, min(1.0, score))
    
    def _calculate_mastery_based_score(self, result: SearchResult,
                                     profile: StudentProfile) -> float:
        """Calculate score based on topic mastery levels"""
        
        # Extract topics from result content
        result_topics = self._extract_topics_from_content(result.content)
        
        if not result_topics:
            return 0.5
        
        mastery_scores = []
        for topic in result_topics:
            mastery = profile.topic_mastery.get(topic, 0.5)
            
            # Prefer content slightly above current mastery level
            if 0.3 <= mastery <= 0.7:  # Learning zone
                mastery_scores.append(0.8)
            elif mastery < 0.3:  # Struggling, need foundational content
                mastery_scores.append(0.6)
            elif mastery > 0.8:  # Mastered, can handle advanced content
                mastery_scores.append(0.4)
            else:
                mastery_scores.append(0.5)
        
        return sum(mastery_scores) / len(mastery_scores)
    
    def _calculate_engagement_prediction(self, content_metrics: ContentMetrics,
                                       profile: StudentProfile) -> float:
        """Predict how engaging this content will be for the student"""
        
        base_engagement = content_metrics.engagement_potential
        
        # Adjust based on learning style
        if profile.learning_style == LearningStyle.VISUAL:
            if 'visual' in content_metrics.__dict__ and content_metrics.diagram_quality > 0.5:
                base_engagement *= 1.2
        elif profile.learning_style == LearningStyle.KINESTHETIC:
            if content_metrics.practical_relevance > 0.7:
                base_engagement *= 1.2
        elif profile.learning_style == LearningStyle.READ_WRITE:
            if content_metrics.explanation_depth > 0.7:
                base_engagement *= 1.1
        
        # Consider past engagement patterns
        base_engagement *= profile.engagement_score
        
        return min(1.0, base_engagement)
    
    def _extract_topics_from_content(self, content: str) -> List[str]:
        """Extract physics topics from content"""
        
        physics_topics = [
            'kinematics', 'dynamics', 'forces', 'energy', 'momentum',
            'thermodynamics', 'waves', 'oscillations', 'electricity',
            'magnetism', 'optics', 'quantum', 'relativity'
        ]
        
        content_lower = content.lower()
        found_topics = [topic for topic in physics_topics if topic in content_lower]
        
        return found_topics
    
    async def _get_personalization_explanation(self, result: SearchResult,
                                             profile: StudentProfile,
                                             context: LearningContext) -> Dict[str, str]:
        """Generate explanation of why content was ranked this way"""
        
        explanations = {}
        
        # Difficulty explanation
        if 'difficulty_level' in result.metadata:
            difficulty = result.metadata['difficulty_level']
            if difficulty == profile.current_level:
                explanations['difficulty'] = f"Matches your {profile.current_level} level"
            else:
                explanations['difficulty'] = f"Content is {difficulty} level"
        
        # Preference explanation
        content_type_preferences = {
            ContentPreference.CONCEPTUAL: "conceptual understanding",
            ContentPreference.PROBLEM_SOLVING: "problem solving practice",
            ContentPreference.MATHEMATICAL: "mathematical formulations",
            ContentPreference.THEORETICAL: "theoretical explanations"
        }
        
        for pref in profile.content_preferences:
            if pref in content_type_preferences:
                explanations['preference'] = f"Matches your preference for {content_type_preferences[pref]}"
                break
        
        # Learning path explanation
        if context.current_topic:
            if context.current_topic.lower() in result.content.lower():
                explanations['learning_path'] = f"Related to your current topic: {context.current_topic}"
        
        # Mastery explanation
        struggling_concepts = [concept for concept in profile.struggling_concepts 
                             if concept.lower() in result.content.lower()]
        if struggling_concepts:
            explanations['mastery'] = f"Addresses concepts you're working on: {', '.join(struggling_concepts)}"
        
        return explanations

# Database setup functions
async def setup_student_profile_tables(postgres_pool: asyncpg.Pool):
    """Setup database tables for student profiles"""
    
    create_tables_sql = """
    CREATE TABLE IF NOT EXISTS student_profiles (
        id SERIAL PRIMARY KEY,
        user_id VARCHAR(255) UNIQUE NOT NULL,
        profile_data JSONB NOT NULL,
        learning_metrics JSONB,
        preferences JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_student_profiles_user_id ON student_profiles(user_id);
    CREATE INDEX IF NOT EXISTS idx_student_profiles_updated ON student_profiles(last_updated);
    
    CREATE TABLE IF NOT EXISTS user_assessments (
        id SERIAL PRIMARY KEY,
        user_id VARCHAR(255) NOT NULL,
        assessment_data JSONB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_user_assessments_user_id ON user_assessments(user_id);
    """
    
    async with postgres_pool.acquire() as conn:
        await conn.execute(create_tables_sql)
    
    logger.info("✅ Student profile tables created")

# Example usage and testing
async def test_context_aware_ranking():
    """Test context-aware ranking system"""
    
    # Mock database setup - in real implementation, use actual connections
    postgres_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'physics_assistant',
        'user': 'physics_user',
        'password': 'physics_secure_password_2024'
    }
    
    try:
        # Create test connections
        postgres_pool = await asyncpg.create_pool(**postgres_config)
        redis_client = redis.Redis(host='localhost', port=6379, password='redis_secure_password_2024', decode_responses=True)
        neo4j_driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "physics_graph_password_2024"))
        
        # Setup tables
        await setup_student_profile_tables(postgres_pool)
        
        # Initialize managers
        profile_manager = StudentProfileManager(postgres_pool, redis_client)
        content_analyzer = ContentAnalyzer(neo4j_driver)
        ranker = ContextAwareRanker(profile_manager, content_analyzer)
        
        # Create test data
        test_user_id = "test_user_123"
        
        # Create test results
        test_results = [
            SearchResult(
                node_id=1, content_type='concept', title='Velocity',
                content='Velocity is the rate of change of position with respect to time',
                similarity_score=0.8, rank=1, metadata={'difficulty_level': 'beginner'}
            ),
            SearchResult(
                node_id=2, content_type='problem', title='Projectile Motion Problem',
                content='Calculate the range of a projectile launched at 45 degrees',
                similarity_score=0.7, rank=2, metadata={'difficulty_level': 'intermediate'}
            ),
            SearchResult(
                node_id=3, content_type='formula', title='Kinetic Energy Formula',
                content='KE = 1/2 * m * v^2 where m is mass and v is velocity',
                similarity_score=0.6, rank=3, metadata={'difficulty_level': 'beginner'}
            )
        ]
        
        # Create test context
        context = LearningContext(
            session_id="test_session",
            user_id=test_user_id,
            current_topic="kinematics",
            session_duration=15,
            previous_queries=["what is velocity", "how to calculate speed"]
        )
        
        # Test ranking
        print("🧪 Testing context-aware ranking...")
        
        ranked_results = await ranker.rank_results(test_results, test_user_id, context)
        
        print(f"📊 Ranked {len(ranked_results)} results:")
        for result in ranked_results:
            print(f"  {result.rank}. {result.title} (score: {result.similarity_score:.3f})")
            print(f"     Type: {result.content_type}")
            if 'personalization_factors' in result.metadata:
                factors = result.metadata['personalization_factors']
                if factors:
                    print(f"     Why: {'; '.join(factors.values())}")
        
        # Test profile update
        print("\n🔄 Testing profile update...")
        
        interaction_data = {
            'success': True,
            'response_time': 25,
            'topic': 'kinematics',
            'concepts': ['velocity', 'acceleration'],
            'session_duration': 20,
            'engagement_score': 0.8,
            'query': 'explain velocity'
        }
        
        await profile_manager.update_student_profile(test_user_id, interaction_data)
        
        # Get updated profile
        profile = await profile_manager.get_student_profile(test_user_id)
        print(f"📋 Updated profile for {test_user_id}:")
        print(f"  Success rate: {profile.success_rate:.2f}")
        print(f"  Engagement: {profile.engagement_score:.2f}")
        print(f"  Topic mastery: {profile.topic_mastery}")
        
        print("\n✅ Context-aware ranking test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        
    finally:
        # Cleanup
        if 'postgres_pool' in locals():
            await postgres_pool.close()
        if 'redis_client' in locals():
            await asyncio.to_thread(redis_client.close)
        if 'neo4j_driver' in locals():
            await asyncio.to_thread(neo4j_driver.close)

if __name__ == "__main__":
    asyncio.run(test_context_aware_ranking())