#!/usr/bin/env python3
"""
Adaptive Learning System for Physics Assistant Phase 6
Implements intelligent tutoring with real-time difficulty adjustment,
knowledge state modeling, and personalized content delivery.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import networkx as nx
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import pickle
import math
import random
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LearningStyle(Enum):
    VISUAL = "visual"
    ANALYTICAL = "analytical"
    KINESTHETIC = "kinesthetic"
    SOCIAL = "social"
    MIXED = "mixed"

class DifficultyLevel(Enum):
    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4

@dataclass
class KnowledgeState:
    """Comprehensive knowledge state representation"""
    student_id: str
    concept_masteries: Dict[str, float]  # Concept -> mastery level [0, 1]
    confidence_levels: Dict[str, float]  # Concept -> confidence [0, 1]
    learning_rates: Dict[str, float]    # Concept -> learning rate
    forgetting_rates: Dict[str, float]  # Concept -> forgetting rate
    misconceptions: Dict[str, List[str]] # Concept -> list of misconceptions
    prerequisite_gaps: List[str]        # Missing prerequisites
    strength_areas: List[str]           # Well-mastered concepts
    growth_areas: List[str]             # Concepts needing work
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class AdaptiveParameters:
    """Adaptive learning parameters for a student"""
    current_difficulty: float           # Current difficulty level [0, 1]
    optimal_difficulty: float          # Optimal difficulty based on performance [0, 1]
    challenge_tolerance: float         # How much challenge student can handle [0, 1]
    learning_velocity: float           # Rate of concept acquisition
    engagement_threshold: float        # Minimum engagement to maintain
    cognitive_load_capacity: float     # Maximum cognitive load
    preferred_learning_style: LearningStyle
    adaptation_sensitivity: float      # How quickly to adapt [0, 1]

@dataclass
class LearningContent:
    """Adaptive learning content recommendation"""
    content_id: str
    content_type: str                  # "problem", "explanation", "example", "practice"
    concept: str
    difficulty: float
    estimated_time: float
    learning_objectives: List[str]
    prerequisites: List[str]
    misconception_addressed: Optional[str] = None
    personalization_score: float = 0.0
    adaptation_reason: str = ""

@dataclass
class LearningSession:
    """Adaptive learning session state"""
    session_id: str
    student_id: str
    start_time: datetime
    current_concept: str
    session_difficulty: float
    completed_activities: List[str]
    performance_history: List[float]
    engagement_indicators: Dict[str, float]
    adaptation_events: List[Dict[str, Any]]
    predicted_outcomes: Dict[str, float]

class BayesianKnowledgeTracer:
    """Bayesian Knowledge Tracing for concept mastery estimation"""
    
    def __init__(self):
        # BKT parameters
        self.p_init = 0.1      # Initial knowledge probability
        self.p_learn = 0.3     # Learning probability
        self.p_guess = 0.25    # Guess probability
        self.p_slip = 0.1      # Slip probability
        
        # Student-specific parameters
        self.student_params = {}
    
    def update_knowledge_state(self, student_id: str, concept: str, 
                             observation: bool, response_time: float = None) -> float:
        """Update knowledge state using Bayesian inference"""
        try:
            # Get current knowledge probability
            if student_id not in self.student_params:
                self.student_params[student_id] = {}
            
            if concept not in self.student_params[student_id]:
                self.student_params[student_id][concept] = {
                    'p_known': self.p_init,
                    'total_attempts': 0,
                    'correct_attempts': 0
                }
            
            params = self.student_params[student_id][concept]
            p_known_prev = params['p_known']
            
            # Update attempt counts
            params['total_attempts'] += 1
            if observation:
                params['correct_attempts'] += 1
            
            # Calculate likelihood of observation given knowledge states
            if observation:
                p_obs_given_known = 1 - self.p_slip
                p_obs_given_unknown = self.p_guess
            else:
                p_obs_given_known = self.p_slip
                p_obs_given_unknown = 1 - self.p_guess
            
            # Bayesian update
            numerator = p_obs_given_known * p_known_prev
            denominator = (p_obs_given_known * p_known_prev + 
                          p_obs_given_unknown * (1 - p_known_prev))
            
            p_known_post = numerator / denominator if denominator > 0 else p_known_prev
            
            # Update with learning probability
            p_known_updated = p_known_post + (1 - p_known_post) * self.p_learn
            
            # Incorporate response time if available
            if response_time is not None:
                time_factor = self._calculate_time_factor(response_time, concept)
                p_known_updated *= time_factor
            
            # Ensure bounds
            params['p_known'] = max(0.01, min(0.99, p_known_updated))
            
            return params['p_known']
            
        except Exception as e:
            logger.error(f"❌ Failed to update knowledge state: {e}")
            return 0.5
    
    def _calculate_time_factor(self, response_time: float, concept: str) -> float:
        """Calculate confidence factor based on response time"""
        # Assume optimal response time varies by concept complexity
        optimal_times = {
            'basic_math': 30,     # seconds
            'kinematics': 120,
            'forces': 180,
            'energy': 150,
            'momentum': 140,
            'angular_motion': 200
        }
        
        optimal_time = optimal_times.get(concept, 120)
        
        # Time factor: 1.0 at optimal time, decreases for very fast/slow responses
        if response_time < optimal_time * 0.3:  # Too fast - might be guessing
            return 0.8
        elif response_time > optimal_time * 3.0:  # Too slow - struggling
            return 0.9
        else:
            # Gaussian-like curve around optimal time
            time_ratio = response_time / optimal_time
            return 1.0 - 0.1 * ((time_ratio - 1.0) ** 2)

class ZoneOfProximalDevelopment:
    """Zone of Proximal Development calculator for optimal challenge"""
    
    def __init__(self):
        self.difficulty_history = defaultdict(list)
        self.performance_history = defaultdict(list)
    
    def calculate_zpd_range(self, student_id: str, concept: str, 
                           current_mastery: float) -> Tuple[float, float]:
        """Calculate optimal difficulty range for student"""
        try:
            # Base ZPD on current mastery level
            base_difficulty = current_mastery
            
            # Adjust based on historical performance
            if student_id in self.performance_history:
                recent_performance = self.performance_history[student_id][-10:]
                if recent_performance:
                    avg_performance = np.mean(recent_performance)
                    performance_trend = self._calculate_trend(recent_performance)
                    
                    # Adjust ZPD based on performance trend
                    if performance_trend > 0.1:  # Improving - can handle more challenge
                        zpd_lower = base_difficulty + 0.1
                        zpd_upper = base_difficulty + 0.3
                    elif performance_trend < -0.1:  # Declining - reduce challenge
                        zpd_lower = max(0.1, base_difficulty - 0.2)
                        zpd_upper = base_difficulty + 0.1
                    else:  # Stable - maintain current level
                        zpd_lower = base_difficulty
                        zpd_upper = base_difficulty + 0.2
                else:
                    zpd_lower = base_difficulty
                    zpd_upper = base_difficulty + 0.2
            else:
                # Default ZPD for new students
                zpd_lower = max(0.1, base_difficulty - 0.1)
                zpd_upper = min(1.0, base_difficulty + 0.2)
            
            # Ensure valid range
            zpd_lower = max(0.1, min(0.9, zpd_lower))
            zpd_upper = max(zpd_lower + 0.1, min(1.0, zpd_upper))
            
            return zpd_lower, zpd_upper
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate ZPD range: {e}")
            return (0.3, 0.7)  # Default range
    
    def update_performance(self, student_id: str, difficulty: float, success: bool):
        """Update performance history for ZPD calculation"""
        self.difficulty_history[student_id].append(difficulty)
        self.performance_history[student_id].append(1.0 if success else 0.0)
        
        # Keep only recent history
        max_history = 50
        if len(self.performance_history[student_id]) > max_history:
            self.performance_history[student_id] = self.performance_history[student_id][-max_history:]
            self.difficulty_history[student_id] = self.difficulty_history[student_id][-max_history:]
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in performance values"""
        if len(values) < 3:
            return 0.0
        
        x = np.arange(len(values))
        z = np.polyfit(x, values, 1)
        return z[0]  # Slope of trend line

class LearningStyleDetector:
    """Detect and adapt to student learning styles"""
    
    def __init__(self):
        self.interaction_patterns = defaultdict(list)
        self.content_preferences = defaultdict(dict)
        self.performance_by_modality = defaultdict(dict)
    
    def detect_learning_style(self, student_id: str, 
                            interaction_history: List[Dict[str, Any]]) -> LearningStyle:
        """Detect primary learning style from interaction patterns"""
        try:
            style_scores = {
                LearningStyle.VISUAL: 0.0,
                LearningStyle.ANALYTICAL: 0.0,
                LearningStyle.KINESTHETIC: 0.0,
                LearningStyle.SOCIAL: 0.0
            }
            
            for interaction in interaction_history:
                content_type = interaction.get('content_type', 'text')
                success = interaction.get('success', False)
                time_spent = interaction.get('time_spent', 0)
                
                # Visual indicators
                if content_type in ['diagram', 'graph', 'animation', 'visualization']:
                    style_scores[LearningStyle.VISUAL] += 2.0 if success else 1.0
                
                # Analytical indicators
                if content_type in ['formula', 'equation', 'proof', 'derivation']:
                    style_scores[LearningStyle.ANALYTICAL] += 2.0 if success else 1.0
                
                # Kinesthetic indicators
                if content_type in ['simulation', 'interactive', 'experiment']:
                    style_scores[LearningStyle.KINESTHETIC] += 2.0 if success else 1.0
                
                # Social indicators
                if interaction.get('help_sought') or interaction.get('collaboration'):
                    style_scores[LearningStyle.SOCIAL] += 1.5 if success else 0.5
                
                # Time-based preferences
                if time_spent > 0:
                    engagement_score = min(2.0, time_spent / 60.0)  # Up to 2 points for 1+ minutes
                    if content_type in ['diagram', 'visualization']:
                        style_scores[LearningStyle.VISUAL] += engagement_score
                    elif content_type in ['text', 'equation']:
                        style_scores[LearningStyle.ANALYTICAL] += engagement_score
            
            # Normalize scores
            total_score = sum(style_scores.values())
            if total_score > 0:
                for style in style_scores:
                    style_scores[style] /= total_score
            
            # Determine primary learning style
            primary_style = max(style_scores, key=style_scores.get)
            
            # If scores are close, classify as mixed
            max_score = style_scores[primary_style]
            second_max = sorted(style_scores.values())[-2]
            
            if max_score - second_max < 0.2:
                return LearningStyle.MIXED
            
            return primary_style
            
        except Exception as e:
            logger.error(f"❌ Failed to detect learning style: {e}")
            return LearningStyle.MIXED

class AdaptiveLearningSystem:
    """Main adaptive learning system coordinating all components"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        
        # Component systems
        self.knowledge_tracer = BayesianKnowledgeTracer()
        self.zpd_calculator = ZoneOfProximalDevelopment()
        self.style_detector = LearningStyleDetector()
        
        # Student models
        self.knowledge_states = {}
        self.adaptive_params = {}
        self.active_sessions = {}
        
        # Content database
        self.content_library = {}
        self.concept_graph = None
        
        # Adaptation configuration
        self.config = {
            'min_mastery_threshold': 0.7,
            'max_difficulty_jump': 0.3,
            'engagement_threshold': 0.6,
            'adaptation_frequency': 5,  # Adapt every N interactions
            'success_rate_target': 0.75,
            'challenge_increase_threshold': 0.85,
            'challenge_decrease_threshold': 0.6
        }
    
    async def initialize(self):
        """Initialize the adaptive learning system"""
        try:
            logger.info("🚀 Initializing Adaptive Learning System")
            
            # Load concept graph
            await self._load_concept_graph()
            
            # Initialize content library
            await self._initialize_content_library()
            
            # Load existing student models
            await self._load_student_models()
            
            logger.info("✅ Adaptive Learning System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Adaptive Learning System: {e}")
            return False
    
    async def _load_concept_graph(self):
        """Load concept graph from database"""
        try:
            if self.db_manager:
                # Load from Neo4j
                concepts_query = """
                MATCH (c:Concept)
                RETURN c.name as name, c.difficulty as difficulty, 
                       c.category as category, c.prerequisites as prerequisites
                """
                
                concepts = await self.db_manager.neo4j.run_query(concepts_query)
                
                self.concept_graph = nx.DiGraph()
                for concept in concepts:
                    self.concept_graph.add_node(
                        concept['name'],
                        difficulty=concept.get('difficulty', 0.5),
                        category=concept.get('category', 'general')
                    )
                
                # Add prerequisite edges
                for concept in concepts:
                    prereqs = concept.get('prerequisites', [])
                    if isinstance(prereqs, str):
                        prereqs = [prereqs]
                    
                    for prereq in prereqs:
                        if self.concept_graph.has_node(prereq):
                            self.concept_graph.add_edge(prereq, concept['name'])
            else:
                # Create sample graph
                self._create_sample_concept_graph()
            
            logger.info(f"📊 Loaded concept graph with {len(self.concept_graph.nodes())} concepts")
            
        except Exception as e:
            logger.error(f"❌ Failed to load concept graph: {e}")
            self._create_sample_concept_graph()
    
    def _create_sample_concept_graph(self):
        """Create sample concept graph for testing"""
        self.concept_graph = nx.DiGraph()
        
        concepts = [
            ("basic_math", {"difficulty": 0.2, "category": "foundation"}),
            ("vectors", {"difficulty": 0.3, "category": "foundation"}),
            ("kinematics_1d", {"difficulty": 0.4, "category": "mechanics"}),
            ("kinematics_2d", {"difficulty": 0.6, "category": "mechanics"}),
            ("forces", {"difficulty": 0.7, "category": "mechanics"}),
            ("energy", {"difficulty": 0.8, "category": "mechanics"}),
            ("momentum", {"difficulty": 0.8, "category": "mechanics"}),
            ("angular_motion", {"difficulty": 0.9, "category": "mechanics"})
        ]
        
        for name, attrs in concepts:
            self.concept_graph.add_node(name, **attrs)
        
        # Add prerequisite relationships
        edges = [
            ("basic_math", "vectors"),
            ("basic_math", "kinematics_1d"),
            ("vectors", "kinematics_2d"),
            ("kinematics_1d", "kinematics_2d"),
            ("kinematics_2d", "forces"),
            ("forces", "energy"),
            ("forces", "momentum"),
            ("momentum", "angular_motion")
        ]
        
        self.concept_graph.add_edges_from(edges)
    
    async def _initialize_content_library(self):
        """Initialize adaptive content library"""
        try:
            # Sample content for different difficulty levels and learning styles
            self.content_library = {
                "basic_math": {
                    0.1: [
                        LearningContent("math_basics_1", "explanation", "basic_math", 0.1, 5, 
                                      ["arithmetic", "algebra"], [], None, 0.9, "foundational")
                    ],
                    0.3: [
                        LearningContent("math_practice_1", "practice", "basic_math", 0.3, 10,
                                      ["problem_solving"], ["arithmetic"], None, 0.8, "skill_building")
                    ]
                },
                "kinematics_1d": {
                    0.4: [
                        LearningContent("kinematics_intro", "explanation", "kinematics_1d", 0.4, 15,
                                      ["position", "velocity", "acceleration"], ["basic_math"], None, 0.9, "conceptual")
                    ],
                    0.6: [
                        LearningContent("kinematics_problems", "problem", "kinematics_1d", 0.6, 20,
                                      ["motion_equations"], ["basic_math"], None, 0.7, "application")
                    ]
                }
            }
            
            logger.info("✅ Content library initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize content library: {e}")
    
    async def _load_student_models(self):
        """Load existing student models from database"""
        try:
            if not self.db_manager:
                return
            
            async with self.db_manager.postgres.get_connection() as conn:
                # Load knowledge states
                students = await conn.fetch("""
                    SELECT DISTINCT user_id FROM interactions 
                    WHERE created_at >= $1
                """, datetime.now() - timedelta(days=30))
                
                for student_row in students:
                    student_id = str(student_row['user_id'])
                    
                    # Initialize knowledge state
                    knowledge_state = await self._initialize_knowledge_state(student_id)
                    self.knowledge_states[student_id] = knowledge_state
                    
                    # Initialize adaptive parameters
                    adaptive_params = await self._initialize_adaptive_parameters(student_id)
                    self.adaptive_params[student_id] = adaptive_params
            
            logger.info(f"👥 Loaded models for {len(self.knowledge_states)} students")
            
        except Exception as e:
            logger.error(f"❌ Failed to load student models: {e}")
    
    async def _initialize_knowledge_state(self, student_id: str) -> KnowledgeState:
        """Initialize knowledge state for a student"""
        try:
            concept_masteries = {}
            confidence_levels = {}
            learning_rates = {}
            forgetting_rates = {}
            misconceptions = {}
            
            if self.concept_graph:
                for concept in self.concept_graph.nodes():
                    # Get initial mastery from database if available
                    if self.db_manager:
                        async with self.db_manager.postgres.get_connection() as conn:
                            progress = await conn.fetchrow("""
                                SELECT proficiency_score FROM user_progress 
                                WHERE user_id = $1 AND topic = $2
                            """, student_id, concept)
                            
                            if progress:
                                concept_masteries[concept] = progress['proficiency_score'] / 100.0
                            else:
                                concept_masteries[concept] = 0.1  # Default low mastery
                    else:
                        concept_masteries[concept] = 0.1
                    
                    # Initialize other parameters
                    confidence_levels[concept] = concept_masteries[concept] * 0.8
                    learning_rates[concept] = 0.3  # Default learning rate
                    forgetting_rates[concept] = 0.05  # Default forgetting rate
                    misconceptions[concept] = []
            
            # Identify prerequisite gaps and strength areas
            prerequisite_gaps = await self._identify_prerequisite_gaps(student_id, concept_masteries)
            strength_areas = [concept for concept, mastery in concept_masteries.items() 
                            if mastery >= self.config['min_mastery_threshold']]
            growth_areas = [concept for concept, mastery in concept_masteries.items() 
                          if mastery < self.config['min_mastery_threshold']]
            
            return KnowledgeState(
                student_id=student_id,
                concept_masteries=concept_masteries,
                confidence_levels=confidence_levels,
                learning_rates=learning_rates,
                forgetting_rates=forgetting_rates,
                misconceptions=misconceptions,
                prerequisite_gaps=prerequisite_gaps,
                strength_areas=strength_areas,
                growth_areas=growth_areas
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize knowledge state for {student_id}: {e}")
            return KnowledgeState(
                student_id=student_id,
                concept_masteries={},
                confidence_levels={},
                learning_rates={},
                forgetting_rates={},
                misconceptions={},
                prerequisite_gaps=[],
                strength_areas=[],
                growth_areas=[]
            )
    
    async def _identify_prerequisite_gaps(self, student_id: str, 
                                        concept_masteries: Dict[str, float]) -> List[str]:
        """Identify missing prerequisites for a student"""
        gaps = []
        
        if not self.concept_graph:
            return gaps
        
        try:
            for concept in concept_masteries:
                if concept in self.concept_graph:
                    # Check if student has mastered prerequisites
                    prerequisites = list(self.concept_graph.predecessors(concept))
                    for prereq in prerequisites:
                        prereq_mastery = concept_masteries.get(prereq, 0.0)
                        if prereq_mastery < self.config['min_mastery_threshold']:
                            if prereq not in gaps:
                                gaps.append(prereq)
            
            return gaps
            
        except Exception as e:
            logger.error(f"❌ Failed to identify prerequisite gaps: {e}")
            return []
    
    async def _initialize_adaptive_parameters(self, student_id: str) -> AdaptiveParameters:
        """Initialize adaptive parameters for a student"""
        try:
            # Get student interaction history to estimate parameters
            current_difficulty = 0.5  # Default medium difficulty
            optimal_difficulty = 0.5
            challenge_tolerance = 0.3
            learning_velocity = 0.5
            engagement_threshold = 0.6
            cognitive_load_capacity = 0.7
            preferred_style = LearningStyle.MIXED
            adaptation_sensitivity = 0.5
            
            if self.db_manager:
                async with self.db_manager.postgres.get_connection() as conn:
                    # Analyze recent interactions to estimate parameters
                    interactions = await conn.fetch("""
                        SELECT agent_type, success, execution_time_ms, metadata
                        FROM interactions 
                        WHERE user_id = $1 AND created_at >= $2
                        ORDER BY created_at DESC
                        LIMIT 20
                    """, student_id, datetime.now() - timedelta(days=7))
                    
                    if interactions:
                        success_rate = sum(1 for i in interactions if i['success']) / len(interactions)
                        avg_response_time = np.mean([i['execution_time_ms'] for i in interactions if i['execution_time_ms']])
                        
                        # Estimate parameters from performance
                        if success_rate > 0.8:
                            challenge_tolerance = 0.6
                            optimal_difficulty = 0.7
                        elif success_rate < 0.5:
                            challenge_tolerance = 0.2
                            optimal_difficulty = 0.3
                        
                        # Estimate learning velocity from success trend
                        successes = [1 if i['success'] else 0 for i in reversed(interactions)]
                        if len(successes) >= 5:
                            trend = np.polyfit(range(len(successes)), successes, 1)[0]
                            learning_velocity = max(0.1, min(1.0, 0.5 + trend))
                        
                        # Detect learning style from interaction patterns
                        interaction_data = []
                        for interaction in interactions:
                            metadata = {}
                            if interaction['metadata']:
                                try:
                                    metadata = json.loads(interaction['metadata'])
                                except:
                                    pass
                            
                            interaction_data.append({
                                'content_type': interaction['agent_type'],
                                'success': interaction['success'],
                                'time_spent': interaction['execution_time_ms'] / 1000.0,
                                'help_sought': metadata.get('help_requested', False)
                            })
                        
                        preferred_style = self.style_detector.detect_learning_style(student_id, interaction_data)
            
            return AdaptiveParameters(
                current_difficulty=current_difficulty,
                optimal_difficulty=optimal_difficulty,
                challenge_tolerance=challenge_tolerance,
                learning_velocity=learning_velocity,
                engagement_threshold=engagement_threshold,
                cognitive_load_capacity=cognitive_load_capacity,
                preferred_learning_style=preferred_style,
                adaptation_sensitivity=adaptation_sensitivity
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize adaptive parameters for {student_id}: {e}")
            return AdaptiveParameters(
                current_difficulty=0.5,
                optimal_difficulty=0.5,
                challenge_tolerance=0.3,
                learning_velocity=0.5,
                engagement_threshold=0.6,
                cognitive_load_capacity=0.7,
                preferred_learning_style=LearningStyle.MIXED,
                adaptation_sensitivity=0.5
            )
    
    async def start_adaptive_session(self, student_id: str, target_concept: str) -> str:
        """Start an adaptive learning session"""
        try:
            session_id = f"session_{student_id}_{datetime.now().timestamp()}"
            
            # Get or create student models
            if student_id not in self.knowledge_states:
                self.knowledge_states[student_id] = await self._initialize_knowledge_state(student_id)
            if student_id not in self.adaptive_params:
                self.adaptive_params[student_id] = await self._initialize_adaptive_parameters(student_id)
            
            knowledge_state = self.knowledge_states[student_id]
            adaptive_params = self.adaptive_params[student_id]
            
            # Determine initial difficulty for the concept
            current_mastery = knowledge_state.concept_masteries.get(target_concept, 0.1)
            zpd_lower, zpd_upper = self.zpd_calculator.calculate_zpd_range(
                student_id, target_concept, current_mastery
            )
            
            # Start with lower end of ZPD
            session_difficulty = zpd_lower
            
            # Create session
            session = LearningSession(
                session_id=session_id,
                student_id=student_id,
                start_time=datetime.now(),
                current_concept=target_concept,
                session_difficulty=session_difficulty,
                completed_activities=[],
                performance_history=[],
                engagement_indicators={},
                adaptation_events=[],
                predicted_outcomes={}
            )
            
            self.active_sessions[session_id] = session
            
            logger.info(f"🎯 Started adaptive session {session_id} for concept {target_concept}")
            return session_id
            
        except Exception as e:
            logger.error(f"❌ Failed to start adaptive session: {e}")
            return ""
    
    async def get_next_content(self, session_id: str) -> Optional[LearningContent]:
        """Get next adaptive content for the session"""
        try:
            if session_id not in self.active_sessions:
                logger.warning(f"⚠️ Session {session_id} not found")
                return None
            
            session = self.active_sessions[session_id]
            student_id = session.student_id
            concept = session.current_concept
            
            knowledge_state = self.knowledge_states[student_id]
            adaptive_params = self.adaptive_params[student_id]
            
            # Determine content difficulty based on current state
            current_mastery = knowledge_state.concept_masteries.get(concept, 0.1)
            session_performance = np.mean(session.performance_history) if session.performance_history else 0.5
            
            # Adapt difficulty based on recent performance
            if len(session.performance_history) >= self.config['adaptation_frequency']:
                session.session_difficulty = await self._adapt_difficulty(
                    session, knowledge_state, adaptive_params
                )
            
            # Select appropriate content
            content = await self._select_content(
                concept, session.session_difficulty, adaptive_params.preferred_learning_style
            )
            
            if content:
                # Personalize content selection
                content.personalization_score = await self._calculate_personalization_score(
                    content, knowledge_state, adaptive_params
                )
                
                # Record adaptation event
                adaptation_event = {
                    'timestamp': datetime.now(),
                    'event_type': 'content_selection',
                    'difficulty_before': session.session_difficulty,
                    'content_difficulty': content.difficulty,
                    'reasoning': content.adaptation_reason
                }
                session.adaptation_events.append(adaptation_event)
            
            return content
            
        except Exception as e:
            logger.error(f"❌ Failed to get next content: {e}")
            return None
    
    async def _adapt_difficulty(self, session: LearningSession, 
                              knowledge_state: KnowledgeState, 
                              adaptive_params: AdaptiveParameters) -> float:
        """Adapt difficulty based on student performance"""
        try:
            current_difficulty = session.session_difficulty
            recent_performance = session.performance_history[-self.config['adaptation_frequency']:]
            avg_performance = np.mean(recent_performance)
            
            # Calculate performance trend
            performance_trend = 0.0
            if len(recent_performance) >= 3:
                x = np.arange(len(recent_performance))
                z = np.polyfit(x, recent_performance, 1)
                performance_trend = z[0]
            
            # Determine difficulty adjustment
            difficulty_adjustment = 0.0
            adaptation_reason = "maintaining_difficulty"
            
            if avg_performance > self.config['challenge_increase_threshold']:
                # Student is performing well - increase challenge
                if performance_trend >= 0:  # And improving or stable
                    difficulty_adjustment = min(
                        self.config['max_difficulty_jump'],
                        adaptive_params.challenge_tolerance * adaptive_params.adaptation_sensitivity
                    )
                    adaptation_reason = "increasing_challenge_high_performance"
            
            elif avg_performance < self.config['challenge_decrease_threshold']:
                # Student is struggling - decrease challenge
                difficulty_adjustment = -min(
                    self.config['max_difficulty_jump'],
                    (1.0 - adaptive_params.challenge_tolerance) * adaptive_params.adaptation_sensitivity
                )
                adaptation_reason = "decreasing_challenge_low_performance"
            
            elif performance_trend < -0.2:
                # Performance declining even if average is okay
                difficulty_adjustment = -0.1 * adaptive_params.adaptation_sensitivity
                adaptation_reason = "decreasing_challenge_declining_trend"
            
            # Apply adjustment
            new_difficulty = current_difficulty + difficulty_adjustment
            
            # Ensure difficulty stays within ZPD
            concept = session.current_concept
            current_mastery = knowledge_state.concept_masteries.get(concept, 0.1)
            zpd_lower, zpd_upper = self.zpd_calculator.calculate_zpd_range(
                session.student_id, concept, current_mastery
            )
            
            new_difficulty = max(zpd_lower, min(zpd_upper, new_difficulty))
            
            # Record adaptation event
            if abs(difficulty_adjustment) > 0.01:
                adaptation_event = {
                    'timestamp': datetime.now(),
                    'event_type': 'difficulty_adaptation',
                    'difficulty_before': current_difficulty,
                    'difficulty_after': new_difficulty,
                    'performance_trigger': avg_performance,
                    'reasoning': adaptation_reason
                }
                session.adaptation_events.append(adaptation_event)
                
                logger.info(f"🎯 Adapted difficulty from {current_difficulty:.2f} to {new_difficulty:.2f} - {adaptation_reason}")
            
            return new_difficulty
            
        except Exception as e:
            logger.error(f"❌ Failed to adapt difficulty: {e}")
            return session.session_difficulty
    
    async def _select_content(self, concept: str, difficulty: float, 
                            learning_style: LearningStyle) -> Optional[LearningContent]:
        """Select appropriate content based on concept, difficulty, and learning style"""
        try:
            # Find content in library that matches criteria
            if concept not in self.content_library:
                # Generate default content
                return LearningContent(
                    content_id=f"default_{concept}_{difficulty:.1f}",
                    content_type="problem",
                    concept=concept,
                    difficulty=difficulty,
                    estimated_time=15.0,
                    learning_objectives=[concept],
                    prerequisites=[],
                    adaptation_reason="default_content_generated"
                )
            
            concept_content = self.content_library[concept]
            
            # Find content with closest difficulty match
            best_content = None
            best_difficulty_match = float('inf')
            
            for content_difficulty, content_list in concept_content.items():
                difficulty_diff = abs(content_difficulty - difficulty)
                if difficulty_diff < best_difficulty_match:
                    best_difficulty_match = difficulty_diff
                    # Select content that matches learning style preference
                    for content in content_list:
                        if self._matches_learning_style(content, learning_style):
                            best_content = content
                            break
                    if not best_content and content_list:
                        best_content = content_list[0]  # Fallback to first available
            
            if best_content:
                # Create a copy with updated difficulty
                return LearningContent(
                    content_id=best_content.content_id,
                    content_type=best_content.content_type,
                    concept=best_content.concept,
                    difficulty=difficulty,  # Use requested difficulty
                    estimated_time=best_content.estimated_time,
                    learning_objectives=best_content.learning_objectives,
                    prerequisites=best_content.prerequisites,
                    misconception_addressed=best_content.misconception_addressed,
                    adaptation_reason="matched_to_student_needs"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to select content: {e}")
            return None
    
    def _matches_learning_style(self, content: LearningContent, style: LearningStyle) -> bool:
        """Check if content matches learning style preference"""
        style_content_mapping = {
            LearningStyle.VISUAL: ['diagram', 'visualization', 'animation'],
            LearningStyle.ANALYTICAL: ['explanation', 'derivation', 'proof'],
            LearningStyle.KINESTHETIC: ['simulation', 'interactive', 'experiment'],
            LearningStyle.SOCIAL: ['collaborative', 'discussion'],
            LearningStyle.MIXED: ['problem', 'practice', 'example']
        }
        
        preferred_types = style_content_mapping.get(style, ['problem'])
        return content.content_type in preferred_types
    
    async def _calculate_personalization_score(self, content: LearningContent,
                                             knowledge_state: KnowledgeState,
                                             adaptive_params: AdaptiveParameters) -> float:
        """Calculate personalization score for content"""
        try:
            score = 0.0
            
            # Difficulty match score
            concept_mastery = knowledge_state.concept_masteries.get(content.concept, 0.1)
            zpd_lower, zpd_upper = self.zpd_calculator.calculate_zpd_range(
                knowledge_state.student_id, content.concept, concept_mastery
            )
            
            if zpd_lower <= content.difficulty <= zpd_upper:
                score += 0.4  # High score for being in ZPD
            else:
                distance_from_zpd = min(
                    abs(content.difficulty - zpd_lower),
                    abs(content.difficulty - zpd_upper)
                )
                score += max(0.0, 0.4 - distance_from_zpd)
            
            # Learning style match score
            if self._matches_learning_style(content, adaptive_params.preferred_learning_style):
                score += 0.3
            
            # Prerequisite readiness score
            prereq_score = 1.0
            for prereq in content.prerequisites:
                prereq_mastery = knowledge_state.concept_masteries.get(prereq, 0.0)
                if prereq_mastery < self.config['min_mastery_threshold']:
                    prereq_score *= prereq_mastery
            score += 0.2 * prereq_score
            
            # Misconception addressing score
            if content.misconception_addressed:
                concept_misconceptions = knowledge_state.misconceptions.get(content.concept, [])
                if content.misconception_addressed in concept_misconceptions:
                    score += 0.1
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate personalization score: {e}")
            return 0.5
    
    async def record_interaction(self, session_id: str, content_id: str, 
                               success: bool, response_time: float, 
                               engagement_indicators: Dict[str, float] = None):
        """Record student interaction and update models"""
        try:
            if session_id not in self.active_sessions:
                logger.warning(f"⚠️ Session {session_id} not found")
                return
            
            session = self.active_sessions[session_id]
            student_id = session.student_id
            concept = session.current_concept
            
            # Update session
            session.completed_activities.append(content_id)
            session.performance_history.append(1.0 if success else 0.0)
            if engagement_indicators:
                session.engagement_indicators.update(engagement_indicators)
            
            # Update knowledge state using Bayesian Knowledge Tracing
            new_mastery = self.knowledge_tracer.update_knowledge_state(
                student_id, concept, success, response_time
            )
            
            # Update student knowledge state
            knowledge_state = self.knowledge_states[student_id]
            knowledge_state.concept_masteries[concept] = new_mastery
            knowledge_state.last_updated = datetime.now()
            
            # Update ZPD calculator
            self.zpd_calculator.update_performance(student_id, session.session_difficulty, success)
            
            # Check for mastery achievement
            if new_mastery >= self.config['min_mastery_threshold']:
                if concept not in knowledge_state.strength_areas:
                    knowledge_state.strength_areas.append(concept)
                if concept in knowledge_state.growth_areas:
                    knowledge_state.growth_areas.remove(concept)
                
                logger.info(f"🎉 Student {student_id} achieved mastery in {concept}")
            
            logger.info(f"📊 Recorded interaction: {concept} - Success: {success} - New mastery: {new_mastery:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Failed to record interaction: {e}")
    
    async def get_student_progress_summary(self, student_id: str) -> Dict[str, Any]:
        """Get comprehensive progress summary for a student"""
        try:
            if student_id not in self.knowledge_states:
                return {}
            
            knowledge_state = self.knowledge_states[student_id]
            adaptive_params = self.adaptive_params.get(student_id)
            
            summary = {
                'student_id': student_id,
                'overall_progress': {
                    'mastered_concepts': len(knowledge_state.strength_areas),
                    'total_concepts': len(knowledge_state.concept_masteries),
                    'mastery_percentage': len(knowledge_state.strength_areas) / len(knowledge_state.concept_masteries) * 100
                    if knowledge_state.concept_masteries else 0,
                    'growth_areas': knowledge_state.growth_areas,
                    'prerequisite_gaps': knowledge_state.prerequisite_gaps
                },
                'knowledge_state': {
                    'concept_masteries': knowledge_state.concept_masteries,
                    'confidence_levels': knowledge_state.confidence_levels,
                    'misconceptions': knowledge_state.misconceptions
                },
                'adaptive_profile': {
                    'learning_style': adaptive_params.preferred_learning_style.value if adaptive_params else 'mixed',
                    'current_difficulty': adaptive_params.current_difficulty if adaptive_params else 0.5,
                    'learning_velocity': adaptive_params.learning_velocity if adaptive_params else 0.5,
                    'challenge_tolerance': adaptive_params.challenge_tolerance if adaptive_params else 0.3
                },
                'recommendations': await self._generate_recommendations(student_id),
                'last_updated': knowledge_state.last_updated
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Failed to get progress summary for {student_id}: {e}")
            return {}
    
    async def _generate_recommendations(self, student_id: str) -> List[str]:
        """Generate personalized recommendations for student"""
        recommendations = []
        
        try:
            knowledge_state = self.knowledge_states[student_id]
            
            # Recommend working on prerequisite gaps
            if knowledge_state.prerequisite_gaps:
                recommendations.append(f"Focus on prerequisite concepts: {', '.join(knowledge_state.prerequisite_gaps[:3])}")
            
            # Recommend growth areas
            if knowledge_state.growth_areas:
                recommendations.append(f"Continue practicing: {', '.join(knowledge_state.growth_areas[:3])}")
            
            # Recommend next concepts based on readiness
            if self.concept_graph:
                ready_concepts = []
                for concept in self.concept_graph.nodes():
                    if concept not in knowledge_state.strength_areas:
                        prerequisites = list(self.concept_graph.predecessors(concept))
                        if all(knowledge_state.concept_masteries.get(prereq, 0.0) >= self.config['min_mastery_threshold'] 
                               for prereq in prerequisites):
                            ready_concepts.append(concept)
                
                if ready_concepts:
                    recommendations.append(f"Ready to learn: {', '.join(ready_concepts[:2])}")
            
            # Add learning style specific recommendations
            adaptive_params = self.adaptive_params.get(student_id)
            if adaptive_params:
                style_recommendations = {
                    LearningStyle.VISUAL: "Try visual aids and diagrams for better understanding",
                    LearningStyle.ANALYTICAL: "Focus on mathematical derivations and logical reasoning",
                    LearningStyle.KINESTHETIC: "Use interactive simulations and hands-on activities",
                    LearningStyle.SOCIAL: "Consider study groups and collaborative learning"
                }
                style_rec = style_recommendations.get(adaptive_params.preferred_learning_style)
                if style_rec:
                    recommendations.append(style_rec)
            
        except Exception as e:
            logger.error(f"❌ Failed to generate recommendations: {e}")
        
        return recommendations

# Testing function
async def test_adaptive_learning_system():
    """Test the adaptive learning system"""
    try:
        logger.info("🧪 Testing Adaptive Learning System")
        
        system = AdaptiveLearningSystem()
        await system.initialize()
        
        # Test session start
        session_id = await system.start_adaptive_session("test_student", "kinematics_1d")
        logger.info(f"✅ Started session: {session_id}")
        
        # Test content generation
        content = await system.get_next_content(session_id)
        if content:
            logger.info(f"✅ Generated content: {content.content_type} - {content.difficulty}")
        
        # Test interaction recording
        await system.record_interaction(session_id, "test_content", True, 30.0)
        logger.info("✅ Recorded interaction")
        
        # Test progress summary
        summary = await system.get_student_progress_summary("test_student")
        logger.info(f"✅ Progress summary: {summary.get('overall_progress', {})}")
        
        logger.info("✅ Adaptive Learning System test completed")
        
    except Exception as e:
        logger.error(f"❌ Adaptive Learning System test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_adaptive_learning_system())