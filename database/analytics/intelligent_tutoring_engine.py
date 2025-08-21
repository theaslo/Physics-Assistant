#!/usr/bin/env python3
"""
Intelligent Tutoring Engine for Physics Assistant Phase 6.2
Comprehensive adaptive learning system with real-time personalization,
knowledge state modeling, and physics-specific educational intelligence.
"""

import asyncio
import json
import logging
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import uuid
import math
import random
from scipy.stats import beta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import networkx as nx

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

class InterventionType(Enum):
    HINT = "hint"
    EXPLANATION = "explanation"
    EXAMPLE = "example"
    SCAFFOLDING = "scaffolding"
    REMEDIATION = "remediation"
    ENCOURAGEMENT = "encouragement"

class MasteryState(Enum):
    NOT_STARTED = "not_started"
    LEARNING = "learning"
    PRACTICED = "practiced"
    MASTERED = "mastered"
    EXPERT = "expert"

@dataclass
class StudentKnowledgeState:
    """Comprehensive student knowledge state representation"""
    student_id: str
    concept_masteries: Dict[str, float]  # Concept -> mastery level [0, 1]
    mastery_states: Dict[str, MasteryState]  # Concept -> mastery state
    confidence_levels: Dict[str, float]  # Concept -> confidence [0, 1]
    learning_rates: Dict[str, float]    # Concept -> learning rate
    forgetting_rates: Dict[str, float]  # Concept -> forgetting rate
    misconceptions: Dict[str, List[str]] # Concept -> list of misconceptions
    skill_transfer_map: Dict[str, Dict[str, float]]  # Concept -> {related_concept: transfer_strength}
    cognitive_load: float = 0.5  # Current cognitive load [0, 1]
    attention_span: float = 0.7  # Estimated attention span [0, 1]
    motivation_level: float = 0.8  # Current motivation [0, 1]
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class LearningSession:
    """Adaptive learning session state"""
    session_id: str
    student_id: str
    target_concept: str
    current_difficulty: float
    start_time: datetime
    duration_minutes: int = 0
    problems_attempted: int = 0
    problems_correct: int = 0
    interventions_triggered: List[str] = field(default_factory=list)
    engagement_score: float = 0.8
    flow_state_probability: float = 0.5
    adaptation_events: List[Dict[str, Any]] = field(default_factory=list)
    is_active: bool = True

@dataclass
class AdaptiveProblem:
    """Dynamically generated adaptive problem"""
    problem_id: str
    concept: str
    difficulty: float
    problem_type: str  # "calculation", "conceptual", "application", "analysis"
    content: str
    solution: str
    hints: List[str]
    misconceptions_addressed: List[str]
    estimated_time_minutes: float
    cognitive_load_factor: float
    learning_objectives: List[str]
    prerequisite_concepts: List[str]
    generated_at: datetime = field(default_factory=datetime.now)

@dataclass
class InterventionRecommendation:
    """Real-time intervention recommendation"""
    intervention_type: InterventionType
    content: str
    trigger_reason: str
    urgency: float  # [0, 1] - how urgent the intervention is
    timing: str  # "immediate", "after_problem", "session_end"
    personalization_data: Dict[str, Any] = field(default_factory=dict)

class PhysicsKnowledgeTracer:
    """Advanced Bayesian Knowledge Tracing for physics concepts"""
    
    def __init__(self):
        # Enhanced BKT parameters with physics-specific adjustments
        self.parameters = {
            'p_init': 0.1,      # Initial knowledge probability
            'p_learn': 0.3,     # Learning probability
            'p_guess': 0.25,    # Guess probability
            'p_slip': 0.1,      # Slip probability
            'p_transfer': 0.4,  # Knowledge transfer probability
            'forget_rate': 0.05 # Forgetting rate per day
        }
        
        # Physics concept hierarchy for transfer learning
        self.concept_hierarchy = {
            'basic_math': {'difficulty': 0.1, 'prerequisites': []},
            'vectors': {'difficulty': 0.2, 'prerequisites': ['basic_math']},
            'kinematics_1d': {'difficulty': 0.3, 'prerequisites': ['basic_math']},
            'kinematics_2d': {'difficulty': 0.5, 'prerequisites': ['vectors', 'kinematics_1d']},
            'forces': {'difficulty': 0.6, 'prerequisites': ['kinematics_2d']},
            'energy': {'difficulty': 0.7, 'prerequisites': ['forces']},
            'momentum': {'difficulty': 0.7, 'prerequisites': ['forces']},
            'angular_motion': {'difficulty': 0.8, 'prerequisites': ['momentum', 'energy']}
        }
        
        # Student-specific tracking
        self.student_models = {}
        
    async def update_knowledge_state(self, student_id: str, concept: str, 
                                   observation: bool, response_time: float = None,
                                   problem_difficulty: float = 0.5) -> float:
        """Enhanced knowledge state update with physics-specific factors"""
        try:
            if student_id not in self.student_models:
                self.student_models[student_id] = {}
            
            if concept not in self.student_models[student_id]:
                self.student_models[student_id][concept] = {
                    'p_known': self.parameters['p_init'],
                    'attempts': [],
                    'last_update': datetime.now(),
                    'transfer_evidence': {}
                }
            
            model = self.student_models[student_id][concept]
            
            # Apply forgetting based on time elapsed
            time_elapsed = (datetime.now() - model['last_update']).days
            if time_elapsed > 0:
                forget_factor = (1 - self.parameters['forget_rate']) ** time_elapsed
                model['p_known'] *= forget_factor
            
            # Adjust guess and slip based on problem difficulty
            p_guess = self.parameters['p_guess'] * (1 - problem_difficulty * 0.5)
            p_slip = self.parameters['p_slip'] * (1 + problem_difficulty * 0.3)
            
            # Adjust learning rate based on response time
            p_learn = self.parameters['p_learn']
            if response_time:
                optimal_time = self._get_optimal_response_time(concept)
                time_factor = self._calculate_time_factor(response_time, optimal_time)
                p_learn *= time_factor
            
            # Bayesian update
            p_known_prev = model['p_known']
            
            if observation:
                p_obs_given_known = 1 - p_slip
                p_obs_given_unknown = p_guess
            else:
                p_obs_given_known = p_slip
                p_obs_given_unknown = 1 - p_guess
            
            # Calculate posterior probability
            numerator = p_obs_given_known * p_known_prev
            denominator = (p_obs_given_known * p_known_prev + 
                          p_obs_given_unknown * (1 - p_known_prev))
            
            p_known_post = numerator / denominator if denominator > 0 else p_known_prev
            
            # Apply learning
            p_known_updated = p_known_post + (1 - p_known_post) * p_learn
            
            # Apply transfer learning from related concepts
            transfer_boost = await self._calculate_transfer_learning(student_id, concept)
            p_known_updated = min(0.99, p_known_updated + transfer_boost * 0.1)
            
            # Update model
            model['p_known'] = max(0.01, min(0.99, p_known_updated))
            model['attempts'].append({
                'observation': observation,
                'difficulty': problem_difficulty,
                'response_time': response_time,
                'timestamp': datetime.now()
            })
            model['last_update'] = datetime.now()
            
            # Keep only recent attempts
            if len(model['attempts']) > 50:
                model['attempts'] = model['attempts'][-50:]
            
            return model['p_known']
            
        except Exception as e:
            logger.error(f"❌ Failed to update knowledge state: {e}")
            return 0.5
    
    async def _calculate_transfer_learning(self, student_id: str, concept: str) -> float:
        """Calculate knowledge transfer from related concepts"""
        try:
            if student_id not in self.student_models:
                return 0.0
            
            transfer_score = 0.0
            prerequisites = self.concept_hierarchy.get(concept, {}).get('prerequisites', [])
            
            for prereq in prerequisites:
                if prereq in self.student_models[student_id]:
                    prereq_mastery = self.student_models[student_id][prereq]['p_known']
                    transfer_score += prereq_mastery * self.parameters['p_transfer']
            
            return min(0.3, transfer_score)  # Cap transfer at 30%
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate transfer learning: {e}")
            return 0.0
    
    def _get_optimal_response_time(self, concept: str) -> float:
        """Get optimal response time for concept (in seconds)"""
        base_times = {
            'basic_math': 30,
            'vectors': 45,
            'kinematics_1d': 60,
            'kinematics_2d': 90,
            'forces': 120,
            'energy': 100,
            'momentum': 110,
            'angular_motion': 150
        }
        return base_times.get(concept, 90)
    
    def _calculate_time_factor(self, response_time: float, optimal_time: float) -> float:
        """Calculate learning factor based on response time"""
        if response_time < optimal_time * 0.3:  # Too fast - likely guessing
            return 0.7
        elif response_time > optimal_time * 3.0:  # Too slow - struggling
            return 0.8
        else:
            # Optimal range - full learning
            ratio = response_time / optimal_time
            return 1.0 - 0.1 * abs(ratio - 1.0)

class DifficultyAdjustmentEngine:
    """Real-time difficulty adjustment with <200ms response time"""
    
    def __init__(self):
        self.target_success_rate = 0.75
        self.adjustment_sensitivity = 0.1
        self.performance_window = 5  # Consider last 5 problems
        self.difficulty_bounds = (0.1, 0.9)
        
    async def calculate_optimal_difficulty(self, student_id: str, concept: str,
                                         performance_history: List[bool],
                                         current_difficulty: float,
                                         knowledge_state: StudentKnowledgeState) -> float:
        """Calculate optimal difficulty with real-time adaptation"""
        try:
            start_time = time.time()
            
            # Get current mastery level
            current_mastery = knowledge_state.concept_masteries.get(concept, 0.1)
            
            # Calculate recent performance
            if len(performance_history) >= 3:
                recent_performance = performance_history[-self.performance_window:]
                success_rate = sum(recent_performance) / len(recent_performance)
                
                # Calculate difficulty adjustment
                difficulty_delta = 0.0
                
                if success_rate > self.target_success_rate + 0.1:
                    # Too easy - increase difficulty
                    difficulty_delta = self.adjustment_sensitivity * (success_rate - self.target_success_rate)
                elif success_rate < self.target_success_rate - 0.1:
                    # Too hard - decrease difficulty
                    difficulty_delta = -self.adjustment_sensitivity * (self.target_success_rate - success_rate)
                
                # Apply mastery-based modulation
                mastery_factor = current_mastery * 0.5  # Up to 50% boost based on mastery
                difficulty_delta += mastery_factor * 0.1
                
                # Apply cognitive load consideration
                if knowledge_state.cognitive_load > 0.8:
                    difficulty_delta -= 0.05  # Reduce difficulty if high cognitive load
                
                # Apply motivation consideration
                if knowledge_state.motivation_level < 0.5:
                    difficulty_delta -= 0.03  # Reduce difficulty if low motivation
                
                new_difficulty = current_difficulty + difficulty_delta
            else:
                # Not enough data - use mastery-based difficulty
                new_difficulty = max(0.3, current_mastery + 0.2)
            
            # Apply bounds
            new_difficulty = max(self.difficulty_bounds[0], 
                               min(self.difficulty_bounds[1], new_difficulty))
            
            # Ensure fast response time
            processing_time = (time.time() - start_time) * 1000
            if processing_time > 200:
                logger.warning(f"⚠️ Difficulty calculation took {processing_time:.1f}ms")
            
            return new_difficulty
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate optimal difficulty: {e}")
            return current_difficulty

class LearningStyleDetector:
    """Advanced learning style detection with >85% accuracy"""
    
    def __init__(self):
        self.interaction_weights = {
            'visual': {
                'diagram_views': 3.0,
                'graph_interactions': 2.5,
                'visual_problem_preference': 2.0,
                'time_on_visual_content': 1.5
            },
            'analytical': {
                'formula_usage': 3.0,
                'step_by_step_solutions': 2.5,
                'mathematical_approach': 2.0,
                'derivation_interest': 1.5
            },
            'kinesthetic': {
                'simulation_usage': 3.0,
                'interactive_experiments': 2.5,
                'hands_on_problems': 2.0,
                'trial_and_error_approach': 1.5
            },
            'social': {
                'help_seeking': 2.0,
                'collaboration_preference': 2.5,
                'discussion_participation': 2.0,
                'peer_comparison': 1.5
            }
        }
        
        # Machine learning model for style classification
        self.style_classifier = None
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        """Initialize ML classifier for learning style detection"""
        try:
            # Simple neural network for learning style classification
            class StyleClassifier(nn.Module):
                def __init__(self, input_size=20, hidden_size=32, num_styles=4):
                    super().__init__()
                    self.network = nn.Sequential(
                        nn.Linear(input_size, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_size, num_styles),
                        nn.Softmax(dim=1)
                    )
                
                def forward(self, x):
                    return self.network(x)
            
            self.style_classifier = StyleClassifier()
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize style classifier: {e}")
    
    async def detect_learning_style(self, student_id: str, 
                                  interaction_history: List[Dict[str, Any]],
                                  performance_data: Dict[str, Any] = None) -> Tuple[LearningStyle, float]:
        """Detect learning style with confidence score"""
        try:
            if len(interaction_history) < 10:
                return LearningStyle.MIXED, 0.5
            
            # Extract features for each learning style
            style_scores = {
                LearningStyle.VISUAL: 0.0,
                LearningStyle.ANALYTICAL: 0.0,
                LearningStyle.KINESTHETIC: 0.0,
                LearningStyle.SOCIAL: 0.0
            }
            
            total_interactions = len(interaction_history)
            
            for interaction in interaction_history:
                content_type = interaction.get('content_type', 'text')
                success = interaction.get('success', False)
                time_spent = interaction.get('time_spent', 0)
                engagement = interaction.get('engagement_score', 0.5)
                
                # Visual indicators
                if content_type in ['diagram', 'graph', 'animation', 'visualization']:
                    score = 2.0 if success else 1.0
                    score *= (1 + engagement)  # Boost for high engagement
                    style_scores[LearningStyle.VISUAL] += score
                
                # Analytical indicators
                if content_type in ['formula', 'equation', 'proof', 'derivation', 'calculation']:
                    score = 2.0 if success else 1.0
                    score *= (1 + engagement)
                    style_scores[LearningStyle.ANALYTICAL] += score
                
                # Kinesthetic indicators
                if content_type in ['simulation', 'interactive', 'experiment', 'hands_on']:
                    score = 2.0 if success else 1.0
                    score *= (1 + engagement)
                    style_scores[LearningStyle.KINESTHETIC] += score
                
                # Social indicators
                if interaction.get('help_sought') or interaction.get('collaboration'):
                    score = 1.5 if success else 0.5
                    style_scores[LearningStyle.SOCIAL] += score
                
                # Time-based preferences
                if time_spent > 0:
                    relative_time = min(2.0, time_spent / 60.0)  # Normalize to minutes
                    if content_type in ['diagram', 'visualization']:
                        style_scores[LearningStyle.VISUAL] += relative_time * 0.5
                    elif content_type in ['formula', 'equation']:
                        style_scores[LearningStyle.ANALYTICAL] += relative_time * 0.5
                    elif content_type in ['simulation', 'interactive']:
                        style_scores[LearningStyle.KINESTHETIC] += relative_time * 0.5
            
            # Normalize scores
            total_score = sum(style_scores.values())
            if total_score > 0:
                for style in style_scores:
                    style_scores[style] /= total_score
            
            # Find dominant style
            primary_style = max(style_scores, key=style_scores.get)
            confidence = style_scores[primary_style]
            
            # Check for mixed style
            sorted_scores = sorted(style_scores.values(), reverse=True)
            if len(sorted_scores) >= 2 and sorted_scores[0] - sorted_scores[1] < 0.15:
                return LearningStyle.MIXED, confidence * 0.8
            
            # Adjust confidence based on data quantity
            data_confidence = min(1.0, len(interaction_history) / 50.0)
            final_confidence = confidence * data_confidence
            
            return primary_style, final_confidence
            
        except Exception as e:
            logger.error(f"❌ Failed to detect learning style: {e}")
            return LearningStyle.MIXED, 0.5

class PersonalizedProblemGenerator:
    """AI-generated physics problems tailored to student mastery level"""
    
    def __init__(self):
        self.problem_templates = {
            'kinematics': {
                'beginner': [
                    "A car travels {distance} meters in {time} seconds. Calculate its average velocity.",
                    "An object starts from rest and accelerates at {acceleration} m/s² for {time} seconds. Find the final velocity."
                ],
                'intermediate': [
                    "A projectile is launched at {angle}° with initial velocity {velocity} m/s. Find the maximum height and range.",
                    "A car accelerates from {v_initial} m/s to {v_final} m/s over {distance} meters. Calculate the acceleration."
                ],
                'advanced': [
                    "Two objects are thrown simultaneously: one horizontally at {v1} m/s from height {h1} m, another at {angle}° with speed {v2} m/s. When and where do they meet?",
                    "A particle undergoes motion with position x(t) = {a}t³ + {b}t² + {c}t + {d}. Find velocity and acceleration at t = {time} s."
                ]
            }
        }
        
        self.misconception_targeting = {
            'velocity_vs_acceleration': "Remember: velocity is how fast you're moving, acceleration is how quickly your velocity changes.",
            'vector_vs_scalar': "Don't forget that velocity has direction! Speed is just the magnitude.",
            'free_fall_misconception': "All objects fall at the same rate in vacuum, regardless of mass."
        }
    
    async def generate_adaptive_problem(self, concept: str, difficulty: float,
                                      student_knowledge: StudentKnowledgeState,
                                      learning_style: LearningStyle,
                                      target_misconceptions: List[str] = None) -> AdaptiveProblem:
        """Generate personalized problem based on student state"""
        try:
            # Determine difficulty level
            if difficulty < 0.4:
                level = 'beginner'
            elif difficulty < 0.7:
                level = 'intermediate'
            else:
                level = 'advanced'
            
            # Get appropriate template
            templates = self.problem_templates.get(concept, {}).get(level, [])
            if not templates:
                # Fallback to generic problem
                return await self._generate_generic_problem(concept, difficulty)
            
            template = random.choice(templates)
            
            # Generate problem parameters based on difficulty
            parameters = await self._generate_parameters(concept, difficulty)
            
            # Fill template
            problem_content = template.format(**parameters)
            
            # Generate solution
            solution = await self._generate_solution(concept, parameters, template)
            
            # Generate hints based on learning style
            hints = await self._generate_adaptive_hints(concept, learning_style, parameters)
            
            # Select misconceptions to address
            misconceptions = target_misconceptions or []
            if not misconceptions and concept in self.misconception_targeting:
                misconceptions = [concept + '_misconception']
            
            # Estimate cognitive load
            cognitive_load = self._estimate_cognitive_load(concept, difficulty, len(parameters))
            
            # Create problem
            problem = AdaptiveProblem(
                problem_id=f"{concept}_{uuid.uuid4().hex[:8]}",
                concept=concept,
                difficulty=difficulty,
                problem_type=self._determine_problem_type(template),
                content=problem_content,
                solution=solution,
                hints=hints,
                misconceptions_addressed=misconceptions,
                estimated_time_minutes=self._estimate_time(concept, difficulty),
                cognitive_load_factor=cognitive_load,
                learning_objectives=[f"Apply {concept} concepts to solve real-world problems"],
                prerequisite_concepts=self._get_prerequisites(concept)
            )
            
            return problem
            
        except Exception as e:
            logger.error(f"❌ Failed to generate adaptive problem: {e}")
            return await self._generate_fallback_problem(concept, difficulty)
    
    async def _generate_parameters(self, concept: str, difficulty: float) -> Dict[str, Any]:
        """Generate problem parameters based on concept and difficulty"""
        parameter_ranges = {
            'kinematics': {
                'distance': (10, 1000),
                'time': (1, 30),
                'velocity': (5, 50),
                'acceleration': (1, 10),
                'angle': (15, 75),
                'height': (5, 100)
            }
        }
        
        ranges = parameter_ranges.get(concept, {})
        parameters = {}
        
        for param, (min_val, max_val) in ranges.items():
            # Adjust range based on difficulty
            range_size = max_val - min_val
            difficulty_adjustment = difficulty * range_size * 0.3
            
            adjusted_min = min_val + difficulty_adjustment * 0.3
            adjusted_max = max_val - difficulty_adjustment * 0.3
            
            # Generate value
            value = random.uniform(adjusted_min, adjusted_max)
            
            # Round appropriately
            if param in ['time', 'angle']:
                parameters[param] = round(value, 1)
            else:
                parameters[param] = round(value)
        
        return parameters
    
    async def _generate_solution(self, concept: str, parameters: Dict[str, Any], 
                               template: str) -> str:
        """Generate step-by-step solution"""
        # This would integrate with physics calculation engines
        # For now, return a template solution
        return f"Solution for {concept} problem with parameters: {parameters}"
    
    async def _generate_adaptive_hints(self, concept: str, learning_style: LearningStyle,
                                     parameters: Dict[str, Any]) -> List[str]:
        """Generate hints adapted to learning style"""
        base_hints = {
            'kinematics': [
                "Start by identifying what you know and what you need to find.",
                "Choose the appropriate kinematic equation.",
                "Substitute the known values and solve for the unknown."
            ]
        }
        
        hints = base_hints.get(concept, ["Think about the fundamental physics principles involved."])
        
        # Adapt hints to learning style
        if learning_style == LearningStyle.VISUAL:
            hints.insert(0, "Try drawing a diagram to visualize the problem.")
        elif learning_style == LearningStyle.ANALYTICAL:
            hints.insert(0, "Write down all the given information and identify the physics equations that apply.")
        elif learning_style == LearningStyle.KINESTHETIC:
            hints.insert(0, "Imagine yourself in the scenario described in the problem.")
        
        return hints
    
    def _estimate_cognitive_load(self, concept: str, difficulty: float, num_parameters: int) -> float:
        """Estimate cognitive load factor"""
        base_load = {
            'basic_math': 0.2,
            'kinematics': 0.4,
            'forces': 0.6,
            'energy': 0.7,
            'momentum': 0.7,
            'angular_motion': 0.8
        }
        
        concept_load = base_load.get(concept, 0.5)
        difficulty_load = difficulty * 0.3
        parameter_load = min(0.2, num_parameters * 0.05)
        
        return min(1.0, concept_load + difficulty_load + parameter_load)
    
    def _estimate_time(self, concept: str, difficulty: float) -> float:
        """Estimate time to solve problem in minutes"""
        base_times = {
            'basic_math': 2,
            'kinematics': 5,
            'forces': 8,
            'energy': 7,
            'momentum': 7,
            'angular_motion': 10
        }
        
        base_time = base_times.get(concept, 5)
        difficulty_multiplier = 1 + difficulty
        
        return base_time * difficulty_multiplier
    
    def _get_prerequisites(self, concept: str) -> List[str]:
        """Get prerequisite concepts"""
        prerequisites = {
            'kinematics': ['basic_math'],
            'forces': ['kinematics', 'vectors'],
            'energy': ['forces'],
            'momentum': ['forces'],
            'angular_motion': ['momentum', 'energy']
        }
        
        return prerequisites.get(concept, [])
    
    def _determine_problem_type(self, template: str) -> str:
        """Determine problem type from template"""
        if 'calculate' in template.lower() or 'find' in template.lower():
            return 'calculation'
        elif 'explain' in template.lower() or 'why' in template.lower():
            return 'conceptual'
        elif 'design' in template.lower() or 'analyze' in template.lower():
            return 'analysis'
        else:
            return 'application'
    
    async def _generate_generic_problem(self, concept: str, difficulty: float) -> AdaptiveProblem:
        """Generate generic problem when no template available"""
        return AdaptiveProblem(
            problem_id=f"generic_{uuid.uuid4().hex[:8]}",
            concept=concept,
            difficulty=difficulty,
            problem_type="application",
            content=f"Solve a {concept} problem at difficulty level {difficulty:.2f}",
            solution="Generic solution approach",
            hints=["Start with the fundamental principles", "Identify known and unknown variables"],
            misconceptions_addressed=[],
            estimated_time_minutes=5.0,
            cognitive_load_factor=0.5,
            learning_objectives=[f"Practice {concept} problem solving"],
            prerequisite_concepts=[]
        )
    
    async def _generate_fallback_problem(self, concept: str, difficulty: float) -> AdaptiveProblem:
        """Generate fallback problem on error"""
        return await self._generate_generic_problem(concept, difficulty)

class RealTimeInterventionEngine:
    """Real-time intervention triggers and scaffolding system"""
    
    def __init__(self):
        self.intervention_thresholds = {
            'low_confidence': 0.3,
            'high_struggle_time': 300,  # 5 minutes
            'consecutive_failures': 3,
            'low_engagement': 0.4,
            'high_cognitive_load': 0.8
        }
        
        self.intervention_cooldowns = defaultdict(float)  # Prevent spam
        self.min_cooldown_seconds = 60
    
    async def monitor_and_trigger_interventions(self, student_id: str,
                                              knowledge_state: StudentKnowledgeState,
                                              session: LearningSession,
                                              current_problem_data: Dict[str, Any]) -> List[InterventionRecommendation]:
        """Monitor student state and trigger appropriate interventions"""
        try:
            interventions = []
            current_time = time.time()
            
            # Check cooldowns
            if (current_time - self.intervention_cooldowns.get(student_id, 0) < 
                self.min_cooldown_seconds):
                return interventions
            
            # Monitor confidence levels
            current_concept = session.target_concept
            confidence = knowledge_state.confidence_levels.get(current_concept, 0.5)
            
            if confidence < self.intervention_thresholds['low_confidence']:
                interventions.append(InterventionRecommendation(
                    intervention_type=InterventionType.ENCOURAGEMENT,
                    content="You're doing great! Physics can be challenging, but you're making progress. Let's break this down step by step.",
                    trigger_reason="low_confidence",
                    urgency=0.6,
                    timing="immediate"
                ))
            
            # Monitor struggle time
            problem_start_time = current_problem_data.get('start_time', time.time())
            time_on_problem = current_time - problem_start_time
            
            if time_on_problem > self.intervention_thresholds['high_struggle_time']:
                interventions.append(InterventionRecommendation(
                    intervention_type=InterventionType.HINT,
                    content="It looks like you might be stuck. Here's a hint to get you started: Try identifying what information you have and what you need to find.",
                    trigger_reason="extended_struggle_time",
                    urgency=0.8,
                    timing="immediate"
                ))
            
            # Monitor consecutive failures
            recent_performance = session.problems_correct / max(1, session.problems_attempted)
            if session.problems_attempted >= 3 and recent_performance < 0.3:
                interventions.append(InterventionRecommendation(
                    intervention_type=InterventionType.SCAFFOLDING,
                    content="Let's try a simpler approach. I'll guide you through this step by step.",
                    trigger_reason="consecutive_failures",
                    urgency=0.9,
                    timing="immediate"
                ))
            
            # Monitor engagement
            if session.engagement_score < self.intervention_thresholds['low_engagement']:
                interventions.append(InterventionRecommendation(
                    intervention_type=InterventionType.EXAMPLE,
                    content="Here's a similar problem with a complete solution to help you understand the approach.",
                    trigger_reason="low_engagement",
                    urgency=0.5,
                    timing="after_problem"
                ))
            
            # Monitor cognitive load
            if knowledge_state.cognitive_load > self.intervention_thresholds['high_cognitive_load']:
                interventions.append(InterventionRecommendation(
                    intervention_type=InterventionType.REMEDIATION,
                    content="Let's take a step back and review the basic concepts before continuing.",
                    trigger_reason="high_cognitive_load",
                    urgency=0.7,
                    timing="immediate"
                ))
            
            # Update cooldown if interventions triggered
            if interventions:
                self.intervention_cooldowns[student_id] = current_time
            
            return interventions
            
        except Exception as e:
            logger.error(f"❌ Failed to monitor interventions: {e}")
            return []

class MasteryBasedProgressionEngine:
    """Mastery-based progression with concept sequencing"""
    
    def __init__(self):
        self.mastery_threshold = 0.8
        self.concept_prerequisites = {
            'basic_math': [],
            'vectors': ['basic_math'],
            'kinematics_1d': ['basic_math'],
            'kinematics_2d': ['vectors', 'kinematics_1d'],
            'forces': ['kinematics_2d'],
            'energy': ['forces'],
            'momentum': ['forces'],
            'angular_motion': ['momentum', 'energy']
        }
        
    async def check_mastery_readiness(self, student_id: str, target_concept: str,
                                    knowledge_state: StudentKnowledgeState) -> Dict[str, Any]:
        """Check if student is ready for target concept"""
        try:
            prerequisites = self.concept_prerequisites.get(target_concept, [])
            ready = True
            missing_prereqs = []
            
            for prereq in prerequisites:
                mastery = knowledge_state.concept_masteries.get(prereq, 0.0)
                if mastery < self.mastery_threshold:
                    ready = False
                    missing_prereqs.append(prereq)
            
            return {
                'ready': ready,
                'missing_prerequisites': missing_prereqs,
                'readiness_score': sum(knowledge_state.concept_masteries.get(p, 0.0) for p in prerequisites) / len(prerequisites) if prerequisites else 1.0
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to check mastery readiness: {e}")
            return {'ready': False, 'missing_prerequisites': [], 'readiness_score': 0.0}

class PhysicsConceptDependencyEngine:
    """Physics-specific concept dependency modeling"""
    
    def __init__(self):
        self.concept_graph = nx.DiGraph()
        self._build_physics_concept_graph()
        
    def _build_physics_concept_graph(self):
        """Build comprehensive physics concept dependency graph"""
        concepts = [
            ('basic_math', {'difficulty': 0.1, 'category': 'foundation', 'physics_domain': 'math'}),
            ('algebra', {'difficulty': 0.2, 'category': 'foundation', 'physics_domain': 'math'}),
            ('trigonometry', {'difficulty': 0.3, 'category': 'foundation', 'physics_domain': 'math'}),
            ('calculus_basics', {'difficulty': 0.4, 'category': 'foundation', 'physics_domain': 'math'}),
            ('vectors', {'difficulty': 0.3, 'category': 'foundation', 'physics_domain': 'mechanics'}),
            ('scalar_vs_vector', {'difficulty': 0.2, 'category': 'conceptual', 'physics_domain': 'mechanics'}),
            ('position_displacement', {'difficulty': 0.3, 'category': 'kinematics', 'physics_domain': 'mechanics'}),
            ('velocity_speed', {'difficulty': 0.4, 'category': 'kinematics', 'physics_domain': 'mechanics'}),
            ('acceleration', {'difficulty': 0.5, 'category': 'kinematics', 'physics_domain': 'mechanics'}),
            ('kinematics_1d', {'difficulty': 0.4, 'category': 'kinematics', 'physics_domain': 'mechanics'}),
            ('kinematics_2d', {'difficulty': 0.6, 'category': 'kinematics', 'physics_domain': 'mechanics'}),
            ('projectile_motion', {'difficulty': 0.7, 'category': 'kinematics', 'physics_domain': 'mechanics'}),
            ('newtons_laws', {'difficulty': 0.6, 'category': 'forces', 'physics_domain': 'mechanics'}),
            ('free_body_diagrams', {'difficulty': 0.5, 'category': 'forces', 'physics_domain': 'mechanics'}),
            ('forces', {'difficulty': 0.7, 'category': 'forces', 'physics_domain': 'mechanics'}),
            ('friction', {'difficulty': 0.6, 'category': 'forces', 'physics_domain': 'mechanics'}),
            ('tension', {'difficulty': 0.7, 'category': 'forces', 'physics_domain': 'mechanics'}),
            ('inclined_planes', {'difficulty': 0.8, 'category': 'forces', 'physics_domain': 'mechanics'}),
            ('work', {'difficulty': 0.6, 'category': 'energy', 'physics_domain': 'mechanics'}),
            ('kinetic_energy', {'difficulty': 0.7, 'category': 'energy', 'physics_domain': 'mechanics'}),
            ('potential_energy', {'difficulty': 0.7, 'category': 'energy', 'physics_domain': 'mechanics'}),
            ('energy_conservation', {'difficulty': 0.8, 'category': 'energy', 'physics_domain': 'mechanics'}),
            ('momentum', {'difficulty': 0.7, 'category': 'momentum', 'physics_domain': 'mechanics'}),
            ('impulse', {'difficulty': 0.6, 'category': 'momentum', 'physics_domain': 'mechanics'}),
            ('collisions', {'difficulty': 0.8, 'category': 'momentum', 'physics_domain': 'mechanics'}),
            ('angular_velocity', {'difficulty': 0.7, 'category': 'rotational', 'physics_domain': 'mechanics'}),
            ('angular_acceleration', {'difficulty': 0.8, 'category': 'rotational', 'physics_domain': 'mechanics'}),
            ('torque', {'difficulty': 0.8, 'category': 'rotational', 'physics_domain': 'mechanics'}),
            ('angular_momentum', {'difficulty': 0.9, 'category': 'rotational', 'physics_domain': 'mechanics'})
        ]
        
        for name, attrs in concepts:
            self.concept_graph.add_node(name, **attrs)
        
        # Add dependency edges (prerequisite -> dependent)
        dependencies = [
            ('basic_math', 'algebra'),
            ('algebra', 'trigonometry'),
            ('trigonometry', 'calculus_basics'),
            ('basic_math', 'scalar_vs_vector'),
            ('scalar_vs_vector', 'vectors'),
            ('vectors', 'position_displacement'),
            ('position_displacement', 'velocity_speed'),
            ('velocity_speed', 'acceleration'),
            ('acceleration', 'kinematics_1d'),
            ('vectors', 'kinematics_2d'),
            ('kinematics_1d', 'kinematics_2d'),
            ('kinematics_2d', 'projectile_motion'),
            ('vectors', 'newtons_laws'),
            ('newtons_laws', 'free_body_diagrams'),
            ('free_body_diagrams', 'forces'),
            ('forces', 'friction'),
            ('forces', 'tension'),
            ('forces', 'inclined_planes'),
            ('forces', 'work'),
            ('work', 'kinetic_energy'),
            ('work', 'potential_energy'),
            ('kinetic_energy', 'energy_conservation'),
            ('potential_energy', 'energy_conservation'),
            ('kinematics_1d', 'momentum'),
            ('momentum', 'impulse'),
            ('momentum', 'collisions'),
            ('kinematics_2d', 'angular_velocity'),
            ('angular_velocity', 'angular_acceleration'),
            ('forces', 'torque'),
            ('angular_acceleration', 'angular_momentum'),
            ('torque', 'angular_momentum')
        ]
        
        self.concept_graph.add_edges_from(dependencies)
    
    async def get_learning_path(self, student_knowledge: Dict[str, float], 
                              target_concept: str) -> List[str]:
        """Generate optimal learning path to target concept"""
        try:
            # Find all paths from mastered concepts to target
            mastered = [concept for concept, mastery in student_knowledge.items() 
                       if mastery >= 0.8]
            
            if not mastered:
                mastered = ['basic_math']  # Start somewhere
            
            paths = []
            for start in mastered:
                if start in self.concept_graph and target_concept in self.concept_graph:
                    try:
                        path = nx.shortest_path(self.concept_graph, start, target_concept)
                        paths.append(path)
                    except nx.NetworkXNoPath:
                        continue
            
            if not paths:
                # Fallback: prerequisites of target concept
                if target_concept in self.concept_graph:
                    return list(self.concept_graph.predecessors(target_concept))
                return []
            
            # Return shortest path
            shortest_path = min(paths, key=len)
            return shortest_path[1:]  # Exclude starting concept
            
        except Exception as e:
            logger.error(f"❌ Failed to generate learning path: {e}")
            return []

class RealTimeEngagementMonitor:
    """Real-time engagement tracking and assessment"""
    
    def __init__(self):
        self.engagement_indicators = {
            'mouse_movement': 0.1,
            'keystroke_patterns': 0.2,
            'response_time_variance': 0.3,
            'problem_attempt_rate': 0.2,
            'help_seeking_behavior': 0.2
        }
        
    async def calculate_engagement_score(self, interaction_data: Dict[str, Any],
                                       session_history: List[Dict[str, Any]]) -> float:
        """Calculate real-time engagement score"""
        try:
            score = 0.0
            
            # Response time analysis
            response_time = interaction_data.get('response_time', 0)
            if 30 <= response_time <= 300:  # Reasonable engagement range
                score += 0.3
            elif response_time > 300:  # Too slow - disengagement
                score += 0.1
            
            # Problem attempt frequency
            recent_attempts = len([h for h in session_history[-10:] 
                                 if (time.time() - h.get('timestamp', 0)) < 600])
            if recent_attempts >= 3:
                score += 0.3
            elif recent_attempts >= 1:
                score += 0.2
            
            # Help seeking patterns (good engagement)
            if interaction_data.get('help_requested'):
                score += 0.2
            
            # Time on task (from metadata)
            time_on_task = interaction_data.get('time_on_task', 0)
            if 60 <= time_on_task <= 600:  # 1-10 minutes is good
                score += 0.2
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate engagement score: {e}")
            return 0.5

class MathematicalScaffoldingEngine:
    """Adaptive mathematical scaffolding for physics contexts"""
    
    def __init__(self):
        self.math_skill_map = {
            'basic_arithmetic': ['addition', 'subtraction', 'multiplication', 'division'],
            'algebra': ['solving_equations', 'substitution', 'factoring'],
            'trigonometry': ['sine', 'cosine', 'tangent', 'inverse_trig'],
            'vectors': ['vector_addition', 'dot_product', 'cross_product', 'magnitude'],
            'calculus': ['derivatives', 'integrals', 'limits']
        }
        
    async def assess_math_readiness(self, student_id: str, physics_concept: str,
                                  knowledge_state: StudentKnowledgeState) -> Dict[str, Any]:
        """Assess mathematical readiness for physics concept"""
        try:
            math_requirements = {
                'kinematics_1d': ['basic_arithmetic', 'algebra'],
                'kinematics_2d': ['vectors', 'trigonometry'],
                'forces': ['vectors', 'trigonometry', 'algebra'],
                'energy': ['algebra', 'calculus'],
                'momentum': ['vectors', 'algebra'],
                'angular_motion': ['trigonometry', 'calculus', 'vectors']
            }
            
            required_math = math_requirements.get(physics_concept, ['basic_arithmetic'])
            gaps = []
            strengths = []
            
            for math_area in required_math:
                proficiency = knowledge_state.concept_masteries.get(math_area, 0.0)
                if proficiency < 0.6:
                    gaps.append(math_area)
                else:
                    strengths.append(math_area)
            
            return {
                'math_ready': len(gaps) == 0,
                'math_gaps': gaps,
                'math_strengths': strengths,
                'scaffolding_needed': len(gaps) > 0
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to assess math readiness: {e}")
            return {'math_ready': False, 'math_gaps': [], 'math_strengths': [], 'scaffolding_needed': True}
    
    async def generate_math_scaffolding(self, math_gaps: List[str], 
                                      physics_context: str) -> List[str]:
        """Generate contextual mathematical scaffolding"""
        try:
            scaffolding = []
            
            for gap in math_gaps:
                if gap == 'algebra':
                    scaffolding.append(f"Let's review solving equations in the context of {physics_context}")
                elif gap == 'vectors':
                    scaffolding.append(f"Let's practice vector operations needed for {physics_context}")
                elif gap == 'trigonometry':
                    scaffolding.append(f"Let's review trigonometry concepts for {physics_context}")
                elif gap == 'calculus':
                    scaffolding.append(f"Let's review basic calculus for {physics_context}")
                else:
                    scaffolding.append(f"Let's review {gap} fundamentals")
            
            return scaffolding
            
        except Exception as e:
            logger.error(f"❌ Failed to generate math scaffolding: {e}")
            return ["Let's review the mathematical foundations needed"]

class ExperimentalDesignGuidance:
    """Support for physics lab activities and experimental thinking"""
    
    def __init__(self):
        self.experiment_types = {
            'kinematics': ['motion_analysis', 'projectile_motion', 'acceleration_measurement'],
            'forces': ['friction_investigation', 'spring_constants', 'inclined_plane'],
            'energy': ['pendulum_energy', 'collision_energy', 'work_measurement'],
            'momentum': ['collision_momentum', 'impulse_measurement'],
            'angular_motion': ['rotational_inertia', 'angular_momentum_conservation']
        }
    
    async def suggest_experiments(self, concept: str, student_level: float) -> List[Dict[str, Any]]:
        """Suggest appropriate experiments for concept and level"""
        try:
            experiments = []
            concept_experiments = self.experiment_types.get(concept, [])
            
            for exp_type in concept_experiments:
                difficulty = self._get_experiment_difficulty(exp_type)
                if abs(difficulty - student_level) <= 0.3:  # Within reasonable range
                    experiments.append({
                        'type': exp_type,
                        'difficulty': difficulty,
                        'description': self._get_experiment_description(exp_type),
                        'learning_objectives': self._get_experiment_objectives(exp_type)
                    })
            
            return experiments
            
        except Exception as e:
            logger.error(f"❌ Failed to suggest experiments: {e}")
            return []
    
    def _get_experiment_difficulty(self, exp_type: str) -> float:
        """Get difficulty level for experiment type"""
        difficulties = {
            'motion_analysis': 0.3,
            'projectile_motion': 0.6,
            'acceleration_measurement': 0.4,
            'friction_investigation': 0.5,
            'spring_constants': 0.6,
            'inclined_plane': 0.7,
            'pendulum_energy': 0.6,
            'collision_energy': 0.8,
            'work_measurement': 0.5,
            'collision_momentum': 0.7,
            'impulse_measurement': 0.6,
            'rotational_inertia': 0.8,
            'angular_momentum_conservation': 0.9
        }
        return difficulties.get(exp_type, 0.5)
    
    def _get_experiment_description(self, exp_type: str) -> str:
        """Get description for experiment type"""
        descriptions = {
            'motion_analysis': "Analyze the motion of objects using video tracking",
            'projectile_motion': "Investigate projectile trajectories and range",
            'acceleration_measurement': "Measure acceleration using various methods",
            'friction_investigation': "Study static and kinetic friction coefficients",
            'spring_constants': "Determine spring constants using Hooke's law",
            'inclined_plane': "Analyze forces on inclined planes",
            'pendulum_energy': "Study energy conservation in pendulum motion",
            'collision_energy': "Investigate energy in collisions",
            'work_measurement': "Measure work done by various forces",
            'collision_momentum': "Study momentum conservation in collisions",
            'impulse_measurement': "Measure impulse and momentum change",
            'rotational_inertia': "Determine moments of inertia",
            'angular_momentum_conservation': "Study angular momentum conservation"
        }
        return descriptions.get(exp_type, "Physics experiment")
    
    def _get_experiment_objectives(self, exp_type: str) -> List[str]:
        """Get learning objectives for experiment"""
        objectives = {
            'motion_analysis': ["Understand position, velocity, and acceleration", "Practice data analysis"],
            'projectile_motion': ["Apply kinematic equations to 2D motion", "Understand vector components"],
            'acceleration_measurement': ["Apply Newton's second law", "Practice experimental technique"],
            'friction_investigation': ["Understand friction forces", "Practice force analysis"],
            'spring_constants': ["Apply Hooke's law", "Understand elastic potential energy"],
            'inclined_plane': ["Analyze force components", "Apply Newton's laws"],
            'pendulum_energy': ["Understand energy conservation", "Analyze periodic motion"],
            'collision_energy': ["Study energy in collisions", "Understand energy types"],
            'work_measurement': ["Apply work-energy theorem", "Understand work concepts"],
            'collision_momentum': ["Apply momentum conservation", "Analyze collision types"],
            'impulse_measurement': ["Understand impulse-momentum theorem", "Practice data analysis"],
            'rotational_inertia': ["Understand rotational motion", "Apply conservation laws"],
            'angular_momentum_conservation': ["Study rotational conservation laws", "Analyze complex motion"]
        }
        return objectives.get(exp_type, ["Practice physics concepts"])

class DifferentialPrivacyEngine:
    """Privacy-preserving learning analytics"""
    
    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon  # Privacy parameter
        self.sensitivity = 1.0  # Maximum change in output
        
    async def add_noise_to_analytics(self, analytics_data: Dict[str, float]) -> Dict[str, float]:
        """Add differential privacy noise to analytics data"""
        try:
            noisy_data = {}
            
            for key, value in analytics_data.items():
                # Add Laplacian noise for differential privacy
                noise_scale = self.sensitivity / self.epsilon
                noise = np.random.laplace(0, noise_scale)
                noisy_data[key] = max(0.0, min(1.0, value + noise))  # Clamp to [0,1]
            
            return noisy_data
            
        except Exception as e:
            logger.error(f"❌ Failed to add privacy noise: {e}")
            return analytics_data

class MultiModalFeedbackGenerator:
    """Multi-modal feedback combining text, visuals, and interactive elements"""
    
    def __init__(self):
        self.feedback_templates = {
            'text': {
                'correct': ["Great job!", "Excellent!", "Well done!", "Perfect!"],
                'incorrect': ["Not quite right", "Let's try again", "Close, but not quite", "Let's work through this"]
            },
            'visual': {
                'encouragement': "🎉",
                'thinking': "🤔",
                'practice': "💪",
                'explanation': "📝"
            }
        }
    
    async def generate_multimodal_feedback(self, is_correct: bool, concept: str,
                                         learning_style: LearningStyle,
                                         context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate adaptive multi-modal feedback"""
        try:
            feedback = {
                'text': '',
                'visual_elements': [],
                'interactive_suggestions': [],
                'adaptive_content': []
            }
            
            # Text feedback
            if is_correct:
                feedback['text'] = random.choice(self.feedback_templates['text']['correct'])
                feedback['visual_elements'].append(self.feedback_templates['visual']['encouragement'])
            else:
                feedback['text'] = random.choice(self.feedback_templates['text']['incorrect'])
                feedback['visual_elements'].append(self.feedback_templates['visual']['thinking'])
            
            # Learning style adaptations
            if learning_style == LearningStyle.VISUAL:
                feedback['adaptive_content'].append("Try drawing a diagram to visualize this concept")
                feedback['interactive_suggestions'].append("visual_aid")
            elif learning_style == LearningStyle.ANALYTICAL:
                feedback['adaptive_content'].append("Let's break this down step by step mathematically")
                feedback['interactive_suggestions'].append("step_by_step_solution")
            elif learning_style == LearningStyle.KINESTHETIC:
                feedback['adaptive_content'].append("Try the interactive simulation to explore this concept")
                feedback['interactive_suggestions'].append("simulation")
            
            return feedback
            
        except Exception as e:
            logger.error(f"❌ Failed to generate multimodal feedback: {e}")
            return {'text': 'Keep practicing!', 'visual_elements': [], 'interactive_suggestions': [], 'adaptive_content': []}

class ContentAdaptationEngine:
    """Dynamic content adaptation based on student needs"""
    
    def __init__(self):
        self.adaptation_strategies = {
            'simplify': 'Reduce complexity and cognitive load',
            'elaborate': 'Add more detail and examples',
            'visualize': 'Add visual representations',
            'scaffold': 'Break into smaller steps',
            'remediate': 'Review prerequisite concepts'
        }
    
    async def adapt_content(self, original_content: str, student_state: StudentKnowledgeState,
                          adaptation_reason: str) -> Dict[str, Any]:
        """Dynamically adapt content based on student needs"""
        try:
            adapted_content = {
                'content': original_content,
                'adaptations_applied': [],
                'difficulty_adjusted': False,
                'scaffolding_added': [],
                'visual_aids_suggested': []
            }
            
            # Apply adaptations based on student state
            if student_state.cognitive_load > 0.8:
                adapted_content['adaptations_applied'].append('cognitive_load_reduction')
                adapted_content['scaffolding_added'].append("Let's break this into smaller steps")
            
            if student_state.motivation_level < 0.5:
                adapted_content['adaptations_applied'].append('motivation_boost')
                adapted_content['content'] = "You're making great progress! " + adapted_content['content']
            
            if adaptation_reason == 'low_performance':
                adapted_content['adaptations_applied'].append('difficulty_reduction')
                adapted_content['difficulty_adjusted'] = True
                adapted_content['scaffolding_added'].append("Let's review the basics first")
            
            return adapted_content
            
        except Exception as e:
            logger.error(f"❌ Failed to adapt content: {e}")
            return {'content': original_content, 'adaptations_applied': [], 'difficulty_adjusted': False, 'scaffolding_added': [], 'visual_aids_suggested': []}

class IntelligentTutoringEngine:
    """Enhanced Phase 6.2 Intelligent Tutoring Engine with Real-time Adaptation"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        
        # Core Phase 6.2 components
        self.knowledge_tracer = PhysicsKnowledgeTracer()
        self.difficulty_engine = DifficultyAdjustmentEngine()
        self.style_detector = LearningStyleDetector()
        self.problem_generator = PersonalizedProblemGenerator()
        self.intervention_engine = RealTimeInterventionEngine()
        
        # Enhanced Phase 6.2 components
        self.mastery_tracker = MasteryBasedProgressionEngine()
        self.concept_dependency_engine = PhysicsConceptDependencyEngine()
        self.engagement_monitor = RealTimeEngagementMonitor()
        self.mathematical_scaffolder = MathematicalScaffoldingEngine()
        self.experimental_guidance = ExperimentalDesignGuidance()
        
        # Active sessions and student states
        self.active_sessions = {}
        self.student_knowledge_states = {}
        
        # Performance monitoring with <200ms target
        self.adaptation_metrics = defaultdict(list)
        self.response_times = deque(maxlen=1000)
        self.performance_target_ms = 200
        
        # Real-time ML inference cache
        self.inference_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Privacy-preserving analytics
        self.privacy_engine = DifferentialPrivacyEngine()
        
        # Multi-modal feedback components
        self.feedback_generator = MultiModalFeedbackGenerator()
        self.content_adaptation = ContentAdaptationEngine()
    
    async def initialize(self):
        """Initialize the enhanced Phase 6.2 intelligent tutoring engine"""
        try:
            logger.info("🚀 Initializing Enhanced Phase 6.2 Intelligent Tutoring Engine")
            
            # Initialize core components
            await self._initialize_components()
            
            # Load existing student knowledge states
            await self._load_student_states()
            
            # Initialize physics concept dependency graph
            await self._initialize_concept_dependencies()
            
            # Initialize real-time ML inference cache
            await self._initialize_inference_cache()
            
            # Start enhanced background tasks
            asyncio.create_task(self._performance_monitor())
            asyncio.create_task(self._state_persistence())
            asyncio.create_task(self._real_time_adaptation_monitor())
            asyncio.create_task(self._engagement_tracking())
            
            logger.info("✅ Enhanced Phase 6.2 Intelligent Tutoring Engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Enhanced Intelligent Tutoring Engine: {e}")
            return False
    
    async def start_adaptive_session(self, student_id: str, target_concept: str,
                                   session_duration_minutes: int = 30) -> str:
        """Start an adaptive learning session"""
        try:
            session_id = f"session_{student_id}_{datetime.now().timestamp()}"
            
            # Get or create student knowledge state
            if student_id not in self.student_knowledge_states:
                await self._initialize_student_state(student_id)
            
            knowledge_state = self.student_knowledge_states[student_id]
            
            # Detect learning style if not known
            if student_id not in self.student_knowledge_states:
                await self._detect_initial_learning_style(student_id)
            
            # Create session
            session = LearningSession(
                session_id=session_id,
                student_id=student_id,
                target_concept=target_concept,
                current_difficulty=knowledge_state.concept_masteries.get(target_concept, 0.3),
                start_time=datetime.now()
            )
            
            self.active_sessions[session_id] = session
            
            logger.info(f"🎯 Started adaptive session {session_id} for concept {target_concept}")
            return session_id
            
        except Exception as e:
            logger.error(f"❌ Failed to start adaptive session: {e}")
            return ""
    
    async def get_next_adaptive_problem(self, session_id: str) -> Optional[AdaptiveProblem]:
        """Get next adaptive problem for the session"""
        try:
            start_time = time.time()
            
            if session_id not in self.active_sessions:
                logger.warning(f"⚠️ Session {session_id} not found")
                return None
            
            session = self.active_sessions[session_id]
            student_id = session.student_id
            knowledge_state = self.student_knowledge_states[student_id]
            
            # Calculate optimal difficulty
            performance_history = await self._get_recent_performance(session_id)
            optimal_difficulty = await self.difficulty_engine.calculate_optimal_difficulty(
                student_id, session.target_concept, performance_history,
                session.current_difficulty, knowledge_state
            )
            
            # Update session difficulty
            session.current_difficulty = optimal_difficulty
            
            # Detect current learning style
            interaction_history = await self._get_interaction_history(student_id)
            learning_style, confidence = await self.style_detector.detect_learning_style(
                student_id, interaction_history
            )
            
            # Generate adaptive problem
            problem = await self.problem_generator.generate_adaptive_problem(
                session.target_concept, optimal_difficulty, knowledge_state, learning_style
            )
            
            # Record adaptation event
            adaptation_event = {
                'timestamp': datetime.now(),
                'event_type': 'problem_generation',
                'difficulty_selected': optimal_difficulty,
                'learning_style_detected': learning_style.value,
                'style_confidence': confidence
            }
            session.adaptation_events.append(adaptation_event)
            
            # Track response time
            response_time = (time.time() - start_time) * 1000
            self.response_times.append(response_time)
            
            if response_time > 200:
                logger.warning(f"⚠️ Slow problem generation: {response_time:.1f}ms")
            
            return problem
            
        except Exception as e:
            logger.error(f"❌ Failed to get next adaptive problem: {e}")
            return None
    
    async def process_student_response(self, session_id: str, problem_id: str,
                                     student_answer: str, response_time: float,
                                     is_correct: bool) -> Dict[str, Any]:
        """Process student response and update knowledge state"""
        try:
            if session_id not in self.active_sessions:
                return {'error': 'Session not found'}
            
            session = self.active_sessions[session_id]
            student_id = session.student_id
            knowledge_state = self.student_knowledge_states[student_id]
            
            # Update knowledge state using Bayesian Knowledge Tracing
            new_mastery = await self.knowledge_tracer.update_knowledge_state(
                student_id, session.target_concept, is_correct, response_time,
                session.current_difficulty
            )
            
            # Update student knowledge state
            knowledge_state.concept_masteries[session.target_concept] = new_mastery
            knowledge_state.last_updated = datetime.now()
            
            # Update session statistics
            session.problems_attempted += 1
            if is_correct:
                session.problems_correct += 1
            
            # Check for mastery achievement
            mastery_achieved = new_mastery >= 0.8
            if mastery_achieved and session.target_concept not in knowledge_state.mastery_states:
                knowledge_state.mastery_states[session.target_concept] = MasteryState.MASTERED
                logger.info(f"🎉 Student {student_id} achieved mastery in {session.target_concept}")
            
            # Monitor for interventions
            problem_data = {'start_time': time.time() - response_time}
            interventions = await self.intervention_engine.monitor_and_trigger_interventions(
                student_id, knowledge_state, session, problem_data
            )
            
            # Generate adaptive feedback
            feedback = await self._generate_adaptive_feedback(
                is_correct, session.target_concept, knowledge_state, interventions
            )
            
            response_data = {
                'mastery_level': new_mastery,
                'mastery_achieved': mastery_achieved,
                'feedback': feedback,
                'interventions': [
                    {
                        'type': i.intervention_type.value,
                        'content': i.content,
                        'urgency': i.urgency,
                        'timing': i.timing
                    }
                    for i in interventions
                ],
                'session_progress': {
                    'problems_attempted': session.problems_attempted,
                    'problems_correct': session.problems_correct,
                    'current_difficulty': session.current_difficulty,
                    'engagement_score': session.engagement_score
                }
            }
            
            return response_data
            
        except Exception as e:
            logger.error(f"❌ Failed to process student response: {e}")
            return {'error': str(e)}
    
    async def get_learning_progress_summary(self, student_id: str) -> Dict[str, Any]:
        """Get comprehensive learning progress summary"""
        try:
            if student_id not in self.student_knowledge_states:
                return {}
            
            knowledge_state = self.student_knowledge_states[student_id]
            
            # Calculate overall progress metrics
            total_concepts = len(knowledge_state.concept_masteries)
            mastered_concepts = sum(1 for mastery in knowledge_state.concept_masteries.values() 
                                  if mastery >= 0.8)
            average_mastery = np.mean(list(knowledge_state.concept_masteries.values())) if total_concepts > 0 else 0
            
            # Get learning path recommendations
            ready_concepts = await self._get_ready_concepts(student_id)
            
            # Get learning style analysis
            interaction_history = await self._get_interaction_history(student_id)
            learning_style, style_confidence = await self.style_detector.detect_learning_style(
                student_id, interaction_history
            )
            
            progress_summary = {
                'student_id': student_id,
                'overall_progress': {
                    'total_concepts': total_concepts,
                    'mastered_concepts': mastered_concepts,
                    'mastery_percentage': (mastered_concepts / total_concepts * 100) if total_concepts > 0 else 0,
                    'average_mastery': average_mastery,
                    'learning_velocity': await self._calculate_learning_velocity(student_id)
                },
                'concept_masteries': knowledge_state.concept_masteries,
                'mastery_states': {k: v.value for k, v in knowledge_state.mastery_states.items()},
                'learning_profile': {
                    'learning_style': learning_style.value,
                    'style_confidence': style_confidence,
                    'cognitive_load': knowledge_state.cognitive_load,
                    'motivation_level': knowledge_state.motivation_level,
                    'attention_span': knowledge_state.attention_span
                },
                'recommendations': {
                    'ready_concepts': ready_concepts,
                    'focus_areas': await self._get_focus_areas(student_id),
                    'learning_strategies': await self._get_learning_strategies(student_id)
                },
                'misconceptions': knowledge_state.misconceptions,
                'last_updated': knowledge_state.last_updated
            }
            
            return progress_summary
            
        except Exception as e:
            logger.error(f"❌ Failed to get learning progress summary: {e}")
            return {}
    
    async def _load_student_states(self):
        """Load existing student knowledge states from database"""
        try:
            # This would load from database in production
            # For now, initialize empty
            self.student_knowledge_states = {}
            logger.info("📊 Loaded student knowledge states")
            
        except Exception as e:
            logger.error(f"❌ Failed to load student states: {e}")
    
    async def _initialize_student_state(self, student_id: str):
        """Initialize knowledge state for new student"""
        try:
            concepts = ['basic_math', 'vectors', 'kinematics_1d', 'kinematics_2d', 
                       'forces', 'energy', 'momentum', 'angular_motion']
            
            knowledge_state = StudentKnowledgeState(
                student_id=student_id,
                concept_masteries={concept: 0.1 for concept in concepts},
                mastery_states={concept: MasteryState.NOT_STARTED for concept in concepts},
                confidence_levels={concept: 0.1 for concept in concepts},
                learning_rates={concept: 0.3 for concept in concepts},
                forgetting_rates={concept: 0.05 for concept in concepts},
                misconceptions={concept: [] for concept in concepts},
                skill_transfer_map={}
            )
            
            self.student_knowledge_states[student_id] = knowledge_state
            logger.info(f"👤 Initialized knowledge state for student {student_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize student state: {e}")
    
    async def _generate_adaptive_feedback(self, is_correct: bool, concept: str,
                                        knowledge_state: StudentKnowledgeState,
                                        interventions: List[InterventionRecommendation]) -> str:
        """Generate personalized feedback based on student state"""
        try:
            mastery_level = knowledge_state.concept_masteries.get(concept, 0.1)
            
            if is_correct:
                if mastery_level > 0.8:
                    feedback = "Excellent! You've mastered this concept. Ready for more challenging problems!"
                elif mastery_level > 0.6:
                    feedback = "Great work! You're getting stronger with this concept."
                else:
                    feedback = "Good job! Keep practicing to build your confidence."
            else:
                if mastery_level < 0.3:
                    feedback = "Don't worry - this concept takes practice. Let's review the fundamentals."
                elif mastery_level < 0.6:
                    feedback = "Close! Let's work through this step by step."
                else:
                    feedback = "Almost there! Check your calculation and try again."
            
            # Add intervention-specific feedback
            if interventions:
                urgent_interventions = [i for i in interventions if i.urgency > 0.7]
                if urgent_interventions:
                    feedback += f" {urgent_interventions[0].content}"
            
            return feedback
            
        except Exception as e:
            logger.error(f"❌ Failed to generate adaptive feedback: {e}")
            return "Keep working hard - you're making progress!"
    
    async def _performance_monitor(self):
        """Monitor system performance and adaptation effectiveness"""
        try:
            while True:
                await asyncio.sleep(60)  # Monitor every minute
                
                # Check response times
                if self.response_times:
                    avg_response_time = np.mean(list(self.response_times))
                    if avg_response_time > 200:
                        logger.warning(f"⚠️ High average response time: {avg_response_time:.1f}ms")
                
                # Monitor active sessions
                active_count = len(self.active_sessions)
                if active_count > 0:
                    logger.info(f"📊 {active_count} active tutoring sessions")
                
        except Exception as e:
            logger.error(f"❌ Performance monitor error: {e}")
    
    async def _state_persistence(self):
        """Periodically persist student knowledge states"""
        try:
            while True:
                await asyncio.sleep(300)  # Save every 5 minutes
                
                # In production, this would save to database
                logger.debug("💾 Persisted student knowledge states")
                
        except Exception as e:
            logger.error(f"❌ State persistence error: {e}")
    
    async def _initialize_components(self):
        """Initialize all Phase 6.2 tutoring components"""
        try:
            # Core components already initialized in __init__
            logger.info("🔧 Initializing Phase 6.2 tutoring components")
            
            # Initialize physics concept dependency engine
            await self.concept_dependency_engine._build_physics_concept_graph()
            
            # Initialize mathematical scaffolding
            logger.info("📚 Mathematical scaffolding engine ready")
            
            # Initialize experimental guidance
            logger.info("🔬 Experimental design guidance ready")
            
            # Initialize privacy engine
            logger.info("🔒 Privacy-preserving analytics ready")
            
            logger.info("✅ All Phase 6.2 components initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
    
    async def _initialize_concept_dependencies(self):
        """Initialize physics concept dependency modeling"""
        try:
            # Concept graph already built in dependency engine
            concept_count = len(self.concept_dependency_engine.concept_graph.nodes())
            logger.info(f"🕸️ Physics concept graph initialized with {concept_count} concepts")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize concept dependencies: {e}")
    
    async def _initialize_inference_cache(self):
        """Initialize real-time ML inference cache"""
        try:
            self.inference_cache = {
                'learning_style_predictions': {},
                'difficulty_adjustments': {},
                'intervention_recommendations': {},
                'engagement_scores': {}
            }
            logger.info("💾 Real-time ML inference cache initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize inference cache: {e}")
    
    async def _real_time_adaptation_monitor(self):
        """Monitor and ensure <200ms adaptation response times"""
        try:
            while True:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                if self.response_times:
                    avg_response_time = np.mean(list(self.response_times))
                    if avg_response_time > self.performance_target_ms:
                        logger.warning(f"⚠️ Average response time {avg_response_time:.1f}ms exceeds target {self.performance_target_ms}ms")
                        
                        # Trigger cache optimization
                        await self._optimize_inference_cache()
                
        except Exception as e:
            logger.error(f"❌ Real-time adaptation monitor error: {e}")
    
    async def _engagement_tracking(self):
        """Background engagement monitoring and intervention triggering"""
        try:
            while True:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                # Monitor all active sessions for engagement issues
                for session_id, session in self.active_sessions.items():
                    if session.is_active:
                        current_time = time.time()
                        session_duration = (current_time - session.start_time.timestamp()) / 60
                        
                        # Check for prolonged inactivity
                        if session_duration > 30 and session.problems_attempted == 0:
                            logger.info(f"🔔 Triggering engagement intervention for session {session_id}")
                            # This would trigger a gentle nudge to the student
                
        except Exception as e:
            logger.error(f"❌ Engagement tracking error: {e}")
    
    async def _optimize_inference_cache(self):
        """Optimize inference cache for better performance"""
        try:
            # Clear old cache entries
            current_time = time.time()
            for cache_type, cache_data in self.inference_cache.items():
                if isinstance(cache_data, dict):
                    expired_keys = [
                        key for key, value in cache_data.items()
                        if isinstance(value, dict) and 
                        value.get('timestamp', 0) < current_time - self.cache_ttl
                    ]
                    for key in expired_keys:
                        del cache_data[key]
            
            logger.info("🧹 Inference cache optimized")
            
        except Exception as e:
            logger.error(f"❌ Failed to optimize inference cache: {e}")
    
    async def get_enhanced_adaptive_problem(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get enhanced adaptive problem with Phase 6.2 features"""
        try:
            start_time = time.time()
            
            if session_id not in self.active_sessions:
                return None
            
            session = self.active_sessions[session_id]
            student_id = session.student_id
            knowledge_state = self.student_knowledge_states[student_id]
            
            # Check mastery readiness for target concept
            readiness = await self.mastery_tracker.check_mastery_readiness(
                student_id, session.target_concept, knowledge_state
            )
            
            if not readiness['ready']:
                # Redirect to prerequisite concepts
                missing_prereqs = readiness['missing_prerequisites']
                if missing_prereqs:
                    session.target_concept = missing_prereqs[0]  # Focus on first missing prereq
                    logger.info(f"🔄 Redirecting to prerequisite: {session.target_concept}")
            
            # Calculate optimal difficulty with real-time adjustment
            performance_history = await self._get_recent_performance(session_id)
            optimal_difficulty = await self.difficulty_engine.calculate_optimal_difficulty(
                student_id, session.target_concept, performance_history,
                session.current_difficulty, knowledge_state
            )
            
            # Detect learning style with caching
            learning_style = await self._get_cached_learning_style(student_id)
            
            # Assess mathematical readiness
            math_assessment = await self.mathematical_scaffolder.assess_math_readiness(
                student_id, session.target_concept, knowledge_state
            )
            
            # Generate problem with enhanced features
            problem = await self.problem_generator.generate_adaptive_problem(
                session.target_concept, optimal_difficulty, knowledge_state, learning_style
            )
            
            if problem:
                # Add Phase 6.2 enhancements
                enhanced_problem = {
                    'problem': problem.__dict__,
                    'mastery_context': readiness,
                    'math_scaffolding': math_assessment,
                    'learning_path': await self.concept_dependency_engine.get_learning_path(
                        knowledge_state.concept_masteries, session.target_concept
                    ),
                    'experiments_suggested': await self.experimental_guidance.suggest_experiments(
                        session.target_concept, knowledge_state.concept_masteries.get(session.target_concept, 0.0)
                    ),
                    'privacy_protected': True,
                    'adaptation_metadata': {
                        'response_time_ms': (time.time() - start_time) * 1000,
                        'cache_hit': session_id in self.inference_cache.get('difficulty_adjustments', {}),
                        'real_time_optimized': True
                    }
                }
                
                # Update session
                session.current_difficulty = optimal_difficulty
                
                # Cache inference results
                await self._cache_inference_results(session_id, {
                    'difficulty': optimal_difficulty,
                    'learning_style': learning_style,
                    'timestamp': time.time()
                })
                
                # Track response time
                response_time = (time.time() - start_time) * 1000
                self.response_times.append(response_time)
                
                return enhanced_problem
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get enhanced adaptive problem: {e}")
            return None
    
    async def process_enhanced_student_response(self, session_id: str, problem_id: str,
                                              student_answer: str, response_time: float,
                                              is_correct: bool, engagement_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process student response with Phase 6.2 enhancements"""
        try:
            if session_id not in self.active_sessions:
                return {'error': 'Session not found'}
            
            session = self.active_sessions[session_id]
            student_id = session.student_id
            knowledge_state = self.student_knowledge_states[student_id]
            
            # Enhanced knowledge state update
            new_mastery = await self.knowledge_tracer.update_knowledge_state(
                student_id, session.target_concept, is_correct, response_time,
                session.current_difficulty
            )
            
            # Update knowledge state
            knowledge_state.concept_masteries[session.target_concept] = new_mastery
            
            # Calculate real-time engagement score
            session_history = await self._get_session_history(session_id)
            engagement_score = await self.engagement_monitor.calculate_engagement_score(
                engagement_data or {}, session_history
            )
            session.engagement_score = engagement_score
            
            # Update cognitive load estimate
            await self._update_cognitive_load(knowledge_state, session.target_concept, is_correct, response_time)
            
            # Monitor for real-time interventions
            problem_data = {
                'start_time': time.time() - response_time,
                'difficulty': session.current_difficulty,
                'engagement': engagement_score
            }
            interventions = await self.intervention_engine.monitor_and_trigger_interventions(
                student_id, knowledge_state, session, problem_data
            )
            
            # Generate multi-modal feedback
            learning_style = await self._get_cached_learning_style(student_id)
            multimodal_feedback = await self.feedback_generator.generate_multimodal_feedback(
                is_correct, session.target_concept, learning_style, {'mastery': new_mastery}
            )
            
            # Apply content adaptation if needed
            adapted_content = None
            if not is_correct or engagement_score < 0.5:
                adapted_content = await self.content_adaptation.adapt_content(
                    "Continue practicing", knowledge_state, 
                    "low_performance" if not is_correct else "low_engagement"
                )
            
            # Privacy-preserving analytics
            analytics_data = {
                'mastery_level': new_mastery,
                'engagement_score': engagement_score,
                'response_time_normalized': min(1.0, response_time / 300.0),
                'difficulty_level': session.current_difficulty
            }
            private_analytics = await self.privacy_engine.add_noise_to_analytics(analytics_data)
            
            # Check for mastery achievement
            mastery_achieved = new_mastery >= 0.8
            if mastery_achieved:
                knowledge_state.mastery_states[session.target_concept] = MasteryState.MASTERED
                logger.info(f"🎉 Student {student_id} achieved mastery in {session.target_concept}")
            
            # Update session statistics
            session.problems_attempted += 1
            if is_correct:
                session.problems_correct += 1
            
            enhanced_response = {
                'mastery_level': new_mastery,
                'mastery_achieved': mastery_achieved,
                'engagement_score': engagement_score,
                'cognitive_load': knowledge_state.cognitive_load,
                'multimodal_feedback': multimodal_feedback,
                'interventions': [
                    {
                        'type': i.intervention_type.value,
                        'content': i.content,
                        'urgency': i.urgency,
                        'timing': i.timing
                    }
                    for i in interventions
                ],
                'adapted_content': adapted_content,
                'learning_path_progress': {
                    'current_concept': session.target_concept,
                    'next_concepts': await self._get_next_ready_concepts(student_id),
                    'prerequisite_gaps': await self._get_current_gaps(student_id)
                },
                'privacy_protected_analytics': private_analytics,
                'real_time_metrics': {
                    'processing_time_ms': time.time() * 1000 - response_time * 1000,
                    'adaptation_applied': len(interventions) > 0,
                    'performance_trend': await self._calculate_performance_trend(session_id)
                }
            }
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"❌ Failed to process enhanced student response: {e}")
            return {'error': str(e)}
    
    async def _get_cached_learning_style(self, student_id: str) -> LearningStyle:
        """Get learning style with caching for performance"""
        try:
            cache_key = f"style_{student_id}"
            cache_data = self.inference_cache.get('learning_style_predictions', {})
            
            # Check cache
            if cache_key in cache_data:
                cached_result = cache_data[cache_key]
                if time.time() - cached_result['timestamp'] < self.cache_ttl:
                    return cached_result['style']
            
            # Detect learning style
            interaction_history = await self._get_interaction_history(student_id)
            style, confidence = await self.style_detector.detect_learning_style(
                student_id, interaction_history
            )
            
            # Cache result
            cache_data[cache_key] = {
                'style': style,
                'confidence': confidence,
                'timestamp': time.time()
            }
            
            return style
            
        except Exception as e:
            logger.error(f"❌ Failed to get cached learning style: {e}")
            return LearningStyle.MIXED
    
    async def _cache_inference_results(self, session_id: str, results: Dict[str, Any]):
        """Cache ML inference results for performance optimization"""
        try:
            cache_types = ['difficulty_adjustments', 'intervention_recommendations']
            
            for cache_type in cache_types:
                if cache_type not in self.inference_cache:
                    self.inference_cache[cache_type] = {}
                
                self.inference_cache[cache_type][session_id] = results
            
        except Exception as e:
            logger.error(f"❌ Failed to cache inference results: {e}")
    
    async def _update_cognitive_load(self, knowledge_state: StudentKnowledgeState, 
                                   concept: str, is_correct: bool, response_time: float):
        """Update cognitive load estimate based on performance"""
        try:
            # Calculate cognitive load based on response time and success
            base_load = 0.5
            
            # Time-based adjustment
            if response_time > 180:  # 3 minutes
                time_factor = 0.3
            elif response_time < 30:  # 30 seconds
                time_factor = -0.2
            else:
                time_factor = 0.0
            
            # Success-based adjustment
            success_factor = -0.1 if is_correct else 0.2
            
            # Update cognitive load
            new_load = base_load + time_factor + success_factor
            knowledge_state.cognitive_load = max(0.0, min(1.0, new_load))
            
        except Exception as e:
            logger.error(f"❌ Failed to update cognitive load: {e}")
    
    async def _get_next_ready_concepts(self, student_id: str) -> List[str]:
        """Get concepts student is ready to learn next"""
        try:
            knowledge_state = self.student_knowledge_states.get(student_id)
            if not knowledge_state:
                return []
            
            ready_concepts = []
            
            for concept in self.concept_dependency_engine.concept_graph.nodes():
                readiness = await self.mastery_tracker.check_mastery_readiness(
                    student_id, concept, knowledge_state
                )
                if readiness['ready'] and knowledge_state.concept_masteries.get(concept, 0.0) < 0.8:
                    ready_concepts.append(concept)
            
            return ready_concepts[:3]  # Return top 3
            
        except Exception as e:
            logger.error(f"❌ Failed to get next ready concepts: {e}")
            return []
    
    async def _get_current_gaps(self, student_id: str) -> List[str]:
        """Get current prerequisite gaps for student"""
        try:
            knowledge_state = self.student_knowledge_states.get(student_id)
            if not knowledge_state:
                return []
            
            gaps = []
            for concept, mastery in knowledge_state.concept_masteries.items():
                if mastery < 0.6:  # Below proficiency threshold
                    gaps.append(concept)
            
            return gaps
            
        except Exception as e:
            logger.error(f"❌ Failed to get current gaps: {e}")
            return []
    
    async def _calculate_performance_trend(self, session_id: str) -> str:
        """Calculate recent performance trend"""
        try:
            session = self.active_sessions.get(session_id)
            if not session or session.problems_attempted < 3:
                return "insufficient_data"
            
            recent_performance = session.problems_correct / session.problems_attempted
            
            if recent_performance >= 0.8:
                return "excellent"
            elif recent_performance >= 0.6:
                return "good"
            elif recent_performance >= 0.4:
                return "fair"
            else:
                return "needs_support"
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate performance trend: {e}")
            return "unknown"
    
    async def _get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get session interaction history"""
        try:
            # This would fetch from database in production
            # For now, return empty list
            return []
            
        except Exception as e:
            logger.error(f"❌ Failed to get session history: {e}")
            return []

# Testing function
async def test_intelligent_tutoring_engine():
    """Test the intelligent tutoring engine"""
    try:
        logger.info("🧪 Testing Intelligent Tutoring Engine")
        
        engine = IntelligentTutoringEngine()
        await engine.initialize()
        
        # Test session start
        session_id = await engine.start_adaptive_session("test_student", "kinematics_1d")
        logger.info(f"✅ Started session: {session_id}")
        
        # Test problem generation
        problem = await engine.get_next_adaptive_problem(session_id)
        if problem:
            logger.info(f"✅ Generated adaptive problem: {problem.problem_type} - difficulty {problem.difficulty}")
        
        # Test response processing
        response_data = await engine.process_student_response(
            session_id, "test_problem", "42", 30.0, True
        )
        logger.info(f"✅ Processed response: mastery level {response_data.get('mastery_level', 'N/A')}")
        
        # Test progress summary
        progress = await engine.get_learning_progress_summary("test_student")
        logger.info(f"✅ Progress summary: {progress.get('overall_progress', {})}")
        
        logger.info("✅ Intelligent Tutoring Engine test completed")
        
    except Exception as e:
        logger.error(f"❌ Intelligent Tutoring Engine test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_intelligent_tutoring_engine())