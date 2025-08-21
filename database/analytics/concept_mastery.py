#!/usr/bin/env python3
"""
Concept Mastery Detection System for Physics Assistant
Advanced algorithms for detecting student concept mastery, analyzing error patterns,
and providing detailed assessments of learning progress.
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy import stats
import networkx as nx
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ErrorPattern:
    """Detailed error pattern analysis"""
    error_type: str
    frequency: int
    concepts_affected: List[str]
    difficulty_correlation: float
    temporal_pattern: str  # 'increasing', 'decreasing', 'stable'
    intervention_priority: str  # 'high', 'medium', 'low'
    description: str
    suggested_remediation: List[str] = field(default_factory=list)

@dataclass
class MasteryEvidence:
    """Evidence supporting mastery assessment"""
    interaction_id: str
    timestamp: datetime
    success: bool
    response_time: float
    difficulty_level: float
    context: Dict[str, Any]
    weight: float = 1.0  # Evidence weight for calculation

@dataclass
class ConceptAssessment:
    """Comprehensive concept mastery assessment"""
    concept_name: str
    mastery_score: float  # 0.0 to 1.0
    confidence_interval: Tuple[float, float]
    evidence_quality: float  # Quality of evidence used
    error_patterns: List[ErrorPattern]
    learning_trajectory: List[float]  # Mastery over time
    prerequisite_status: Dict[str, float]
    next_steps: List[str]
    assessment_timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class MisconcepationPattern:
    """Identified misconception pattern"""
    misconception_id: str
    description: str
    affected_concepts: List[str]
    manifestation_frequency: float
    severity_score: float  # Impact on learning
    typical_errors: List[str]
    corrective_strategies: List[str]

class ConceptMasteryDetector:
    """Advanced concept mastery detection system"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.mastery_models = {}
        self.error_taxonomies = {}
        self.misconception_patterns = []
        
        # Configuration for mastery detection
        self.config = {
            'mastery_threshold': 0.75,
            'evidence_window_days': 14,
            'min_evidence_count': 5,
            'confidence_threshold': 0.8,
            'error_clustering_eps': 0.3,
            'temporal_weight_decay': 0.95,
            'difficulty_weight_factor': 1.2,
            'consistency_requirement': 0.7
        }
        
        # Initialize error taxonomies
        self._initialize_error_taxonomies()
        self._initialize_misconception_patterns()
    
    def _initialize_error_taxonomies(self):
        """Initialize physics-specific error taxonomies"""
        self.error_taxonomies = {
            'kinematics': {
                'conceptual_errors': [
                    'velocity_acceleration_confusion',
                    'vector_scalar_confusion',
                    'reference_frame_errors',
                    'sign_convention_errors'
                ],
                'procedural_errors': [
                    'equation_selection_errors',
                    'algebraic_manipulation_errors',
                    'unit_conversion_errors',
                    'calculation_errors'
                ],
                'representational_errors': [
                    'graph_interpretation_errors',
                    'diagram_analysis_errors',
                    'symbolic_representation_errors'
                ]
            },
            'forces': {
                'conceptual_errors': [
                    'force_motion_confusion',
                    'normal_force_misconceptions',
                    'friction_understanding_errors',
                    'action_reaction_confusion'
                ],
                'procedural_errors': [
                    'free_body_diagram_errors',
                    'component_resolution_errors',
                    'equilibrium_analysis_errors',
                    'newton_law_application_errors'
                ],
                'representational_errors': [
                    'vector_diagram_errors',
                    'force_magnitude_estimation_errors',
                    'coordinate_system_errors'
                ]
            },
            'energy': {
                'conceptual_errors': [
                    'energy_conservation_violations',
                    'kinetic_potential_confusion',
                    'work_energy_relationship_errors',
                    'power_energy_confusion'
                ],
                'procedural_errors': [
                    'energy_calculation_errors',
                    'work_calculation_errors',
                    'reference_level_errors',
                    'efficiency_calculation_errors'
                ],
                'representational_errors': [
                    'energy_diagram_errors',
                    'work_path_dependency_errors',
                    'energy_transformation_errors'
                ]
            }
        }
    
    def _initialize_misconception_patterns(self):
        """Initialize common physics misconceptions"""
        self.misconception_patterns = [
            MisconcepationPattern(
                misconception_id="force_implies_motion",
                description="Belief that force always implies motion",
                affected_concepts=["forces", "equilibrium", "friction"],
                manifestation_frequency=0.3,
                severity_score=0.8,
                typical_errors=["assuming_motion_with_applied_force", "ignoring_static_friction"],
                corrective_strategies=["static_equilibrium_examples", "force_balance_practice"]
            ),
            MisconcepationPattern(
                misconception_id="velocity_acceleration_conflation",
                description="Confusing velocity and acceleration concepts",
                affected_concepts=["kinematics", "forces"],
                manifestation_frequency=0.4,
                severity_score=0.9,
                typical_errors=["using_velocity_for_acceleration", "direction_confusion"],
                corrective_strategies=["velocity_acceleration_comparison", "graphical_representations"]
            ),
            MisconcepationPattern(
                misconception_id="energy_disappears",
                description="Belief that energy can disappear or be lost",
                affected_concepts=["energy", "momentum"],
                manifestation_frequency=0.25,
                severity_score=0.7,
                typical_errors=["ignoring_energy_transformation", "missing_dissipative_forces"],
                corrective_strategies=["energy_transformation_tracking", "dissipation_examples"]
            )
        ]
    
    async def assess_concept_mastery(self, user_id: str, concept: str, 
                                   evidence_window_days: int = None) -> ConceptAssessment:
        """Comprehensive concept mastery assessment"""
        try:
            window_days = evidence_window_days or self.config['evidence_window_days']
            
            # Collect evidence
            evidence = await self._collect_mastery_evidence(user_id, concept, window_days)
            
            if len(evidence) < self.config['min_evidence_count']:
                return ConceptAssessment(
                    concept_name=concept,
                    mastery_score=0.0,
                    confidence_interval=(0.0, 0.0),
                    evidence_quality=0.0,
                    error_patterns=[],
                    learning_trajectory=[],
                    prerequisite_status={},
                    next_steps=["Increase practice frequency", "Need more interaction data"]
                )
            
            # Calculate mastery score
            mastery_score, confidence_interval = self._calculate_mastery_score(evidence)
            
            # Analyze error patterns
            error_patterns = await self._analyze_error_patterns(user_id, concept, evidence)
            
            # Calculate learning trajectory
            learning_trajectory = self._calculate_learning_trajectory(evidence)
            
            # Check prerequisite status
            prerequisite_status = await self._assess_prerequisite_status(user_id, concept)
            
            # Generate next steps
            next_steps = self._generate_next_steps(mastery_score, error_patterns, prerequisite_status)
            
            # Calculate evidence quality
            evidence_quality = self._assess_evidence_quality(evidence)
            
            return ConceptAssessment(
                concept_name=concept,
                mastery_score=mastery_score,
                confidence_interval=confidence_interval,
                evidence_quality=evidence_quality,
                error_patterns=error_patterns,
                learning_trajectory=learning_trajectory,
                prerequisite_status=prerequisite_status,
                next_steps=next_steps
            )
        
        except Exception as e:
            logger.error(f"❌ Failed to assess concept mastery for {concept}: {e}")
            return ConceptAssessment(
                concept_name=concept,
                mastery_score=0.0,
                confidence_interval=(0.0, 0.0),
                evidence_quality=0.0,
                error_patterns=[],
                learning_trajectory=[],
                prerequisite_status={},
                next_steps=["Assessment failed - check data availability"]
            )
    
    async def _collect_mastery_evidence(self, user_id: str, concept: str, 
                                      window_days: int) -> List[MasteryEvidence]:
        """Collect evidence for mastery assessment"""
        evidence = []
        
        if not self.db_manager:
            return evidence
        
        try:
            since_date = datetime.now() - timedelta(days=window_days)
            
            async with self.db_manager.postgres.get_connection() as conn:
                interactions = await conn.fetch("""
                    SELECT id, success, created_at, execution_time_ms, 
                           request_data, response_data, metadata
                    FROM interactions 
                    WHERE user_id = $1 AND agent_type = $2 AND created_at >= $3
                    ORDER BY created_at ASC
                """, user_id, concept, since_date)
            
            for interaction in interactions:
                # Parse metadata for difficulty and context
                try:
                    metadata = json.loads(interaction['metadata']) if isinstance(interaction['metadata'], str) else interaction['metadata']
                    difficulty_level = metadata.get('difficulty_level', 1.0)
                    context = metadata.get('context', {})
                except:
                    difficulty_level = 1.0
                    context = {}
                
                # Calculate evidence weight based on recency and difficulty
                days_ago = (datetime.now() - interaction['created_at']).days
                temporal_weight = self.config['temporal_weight_decay'] ** days_ago
                difficulty_weight = difficulty_level * self.config['difficulty_weight_factor']
                
                evidence_item = MasteryEvidence(
                    interaction_id=str(interaction['id']),
                    timestamp=interaction['created_at'],
                    success=interaction['success'],
                    response_time=interaction['execution_time_ms'] / 1000.0 if interaction['execution_time_ms'] else 0.0,
                    difficulty_level=difficulty_level,
                    context=context,
                    weight=temporal_weight * difficulty_weight
                )
                
                evidence.append(evidence_item)
            
            return evidence
        
        except Exception as e:
            logger.error(f"❌ Failed to collect mastery evidence: {e}")
            return []
    
    def _calculate_mastery_score(self, evidence: List[MasteryEvidence]) -> Tuple[float, Tuple[float, float]]:
        """Calculate mastery score with confidence interval"""
        try:
            # Weighted success rate
            successes = [e.success for e in evidence]
            weights = [e.weight for e in evidence]
            
            weighted_success_rate = np.average(successes, weights=weights)
            
            # Calculate confidence interval using bootstrap
            n_bootstrap = 1000
            bootstrap_scores = []
            
            for _ in range(n_bootstrap):
                # Bootstrap sample with replacement
                sample_indices = np.random.choice(len(evidence), size=len(evidence), replace=True)
                sample_successes = [successes[i] for i in sample_indices]
                sample_weights = [weights[i] for i in sample_indices]
                
                if sum(sample_weights) > 0:
                    score = np.average(sample_successes, weights=sample_weights)
                    bootstrap_scores.append(score)
            
            # Calculate confidence interval
            if bootstrap_scores:
                confidence_interval = (
                    np.percentile(bootstrap_scores, 2.5),
                    np.percentile(bootstrap_scores, 97.5)
                )
            else:
                confidence_interval = (weighted_success_rate, weighted_success_rate)
            
            # Apply consistency penalty
            if len(evidence) >= 5:
                recent_evidence = evidence[-5:]
                recent_successes = [e.success for e in recent_evidence]
                consistency = 1.0 - np.std(recent_successes)
                
                if consistency < self.config['consistency_requirement']:
                    penalty = (self.config['consistency_requirement'] - consistency) * 0.5
                    weighted_success_rate = max(0.0, weighted_success_rate - penalty)
            
            return weighted_success_rate, confidence_interval
        
        except Exception as e:
            logger.error(f"❌ Failed to calculate mastery score: {e}")
            return 0.0, (0.0, 0.0)
    
    async def _analyze_error_patterns(self, user_id: str, concept: str, 
                                    evidence: List[MasteryEvidence]) -> List[ErrorPattern]:
        """Analyze error patterns in student interactions"""
        error_patterns = []
        
        try:
            # Collect failed interactions for error analysis
            failed_evidence = [e for e in evidence if not e.success]
            
            if len(failed_evidence) < 2:
                return error_patterns
            
            # Get detailed error information from database
            if self.db_manager:
                interaction_ids = [e.interaction_id for e in failed_evidence]
                
                async with self.db_manager.postgres.get_connection() as conn:
                    error_details = await conn.fetch("""
                        SELECT interaction_id, error_message, metadata, request_data, response_data
                        FROM interactions 
                        WHERE id = ANY($1)
                    """, interaction_ids)
                
                # Analyze error types and patterns
                error_type_counts = Counter()
                temporal_patterns = defaultdict(list)
                
                for error_detail in error_details:
                    # Extract error types from various sources
                    error_types = self._extract_error_types(
                        error_detail['error_message'],
                        error_detail['metadata'],
                        error_detail['request_data'],
                        error_detail['response_data'],
                        concept
                    )
                    
                    for error_type in error_types:
                        error_type_counts[error_type] += 1
                    
                    # Track temporal patterns
                    interaction_evidence = next((e for e in failed_evidence 
                                               if e.interaction_id == str(error_detail['interaction_id'])), None)
                    if interaction_evidence:
                        temporal_patterns[error_type].append(interaction_evidence.timestamp)
                
                # Create error pattern objects
                for error_type, frequency in error_type_counts.items():
                    if frequency >= 2:  # Minimum frequency threshold
                        temporal_trend = self._analyze_temporal_trend(temporal_patterns[error_type])
                        
                        error_pattern = ErrorPattern(
                            error_type=error_type,
                            frequency=frequency,
                            concepts_affected=[concept],
                            difficulty_correlation=self._calculate_difficulty_correlation(error_type, evidence),
                            temporal_pattern=temporal_trend,
                            intervention_priority=self._determine_intervention_priority(error_type, frequency, temporal_trend),
                            description=self._get_error_description(error_type, concept),
                            suggested_remediation=self._get_remediation_strategies(error_type, concept)
                        )
                        
                        error_patterns.append(error_pattern)
            
            return error_patterns
        
        except Exception as e:
            logger.error(f"❌ Failed to analyze error patterns: {e}")
            return []
    
    def _extract_error_types(self, error_message: str, metadata: Any, 
                           request_data: Any, response_data: Any, concept: str) -> List[str]:
        """Extract error types from interaction data"""
        error_types = []
        
        try:
            # Parse metadata
            if metadata:
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                if 'error_type' in metadata:
                    error_types.append(metadata['error_type'])
                if 'error_category' in metadata:
                    error_types.append(metadata['error_category'])
            
            # Analyze error message
            if error_message:
                error_types.extend(self._classify_error_from_message(error_message, concept))
            
            # Analyze request/response patterns
            if request_data and response_data:
                pattern_errors = self._identify_pattern_errors(request_data, response_data, concept)
                error_types.extend(pattern_errors)
            
            # Map to standardized error taxonomy
            standardized_errors = []
            concept_taxonomy = self.error_taxonomies.get(concept, {})
            
            for error_type in error_types:
                for category, error_list in concept_taxonomy.items():
                    if any(keyword in error_type.lower() for keyword in error_list):
                        standardized_errors.append(f"{category}:{error_type}")
                        break
                else:
                    standardized_errors.append(f"unclassified:{error_type}")
            
            return list(set(standardized_errors))  # Remove duplicates
        
        except Exception as e:
            logger.error(f"❌ Failed to extract error types: {e}")
            return ['analysis_error']
    
    def _classify_error_from_message(self, error_message: str, concept: str) -> List[str]:
        """Classify error based on error message content"""
        error_types = []
        message_lower = error_message.lower()
        
        # Common error patterns
        if any(word in message_lower for word in ['unit', 'dimension', 'conversion']):
            error_types.append('unit_error')
        
        if any(word in message_lower for word in ['sign', 'positive', 'negative', 'direction']):
            error_types.append('sign_error')
        
        if any(word in message_lower for word in ['magnitude', 'value', 'calculation']):
            error_types.append('calculation_error')
        
        if any(word in message_lower for word in ['equation', 'formula', 'method']):
            error_types.append('method_selection_error')
        
        if any(word in message_lower for word in ['concept', 'understanding', 'definition']):
            error_types.append('conceptual_error')
        
        return error_types
    
    def _identify_pattern_errors(self, request_data: Any, response_data: Any, concept: str) -> List[str]:
        """Identify errors from request/response patterns"""
        pattern_errors = []
        
        try:
            # Parse request and response data
            if isinstance(request_data, str):
                request_data = json.loads(request_data)
            if isinstance(response_data, str):
                response_data = json.loads(response_data)
            
            # Check for common patterns in physics problems
            if concept == 'kinematics':
                # Check for velocity/acceleration confusion
                if 'velocity' in str(request_data).lower() and 'acceleration' in str(response_data).lower():
                    pattern_errors.append('velocity_acceleration_confusion')
            
            elif concept == 'forces':
                # Check for force/motion confusion
                if 'force' in str(request_data).lower() and 'motion' in str(response_data).lower():
                    pattern_errors.append('force_motion_confusion')
            
            elif concept == 'energy':
                # Check for energy conservation errors
                if 'conservation' in str(request_data).lower():
                    pattern_errors.append('conservation_error')
        
        except Exception as e:
            logger.error(f"❌ Failed to identify pattern errors: {e}")
        
        return pattern_errors
    
    def _analyze_temporal_trend(self, timestamps: List[datetime]) -> str:
        """Analyze temporal trend of error occurrences"""
        if len(timestamps) < 3:
            return 'insufficient_data'
        
        # Sort timestamps
        timestamps_sorted = sorted(timestamps)
        
        # Calculate time intervals
        intervals = [(timestamps_sorted[i+1] - timestamps_sorted[i]).total_seconds() 
                    for i in range(len(timestamps_sorted) - 1)]
        
        # Determine trend
        if len(intervals) >= 2:
            recent_avg = np.mean(intervals[-2:]) if len(intervals) >= 2 else intervals[-1]
            earlier_avg = np.mean(intervals[:-2]) if len(intervals) > 2 else intervals[0]
            
            if recent_avg < earlier_avg * 0.8:
                return 'increasing'  # Errors happening more frequently
            elif recent_avg > earlier_avg * 1.2:
                return 'decreasing'  # Errors happening less frequently
            else:
                return 'stable'
        
        return 'stable'
    
    def _calculate_difficulty_correlation(self, error_type: str, evidence: List[MasteryEvidence]) -> float:
        """Calculate correlation between error type and difficulty level"""
        try:
            # Find evidence with this error type (simplified)
            error_difficulties = []
            success_difficulties = []
            
            for e in evidence:
                if e.success:
                    success_difficulties.append(e.difficulty_level)
                else:
                    error_difficulties.append(e.difficulty_level)
            
            if error_difficulties and success_difficulties:
                error_avg = np.mean(error_difficulties)
                success_avg = np.mean(success_difficulties)
                
                # Correlation is higher error difficulty relative to success difficulty
                correlation = (error_avg - success_avg) / max(success_avg, 1.0)
                return max(0.0, min(1.0, correlation))
            
            return 0.5  # Neutral correlation
        
        except Exception as e:
            logger.error(f"❌ Failed to calculate difficulty correlation: {e}")
            return 0.5
    
    def _determine_intervention_priority(self, error_type: str, frequency: int, temporal_trend: str) -> str:
        """Determine intervention priority for error pattern"""
        # Base priority on frequency
        if frequency >= 5:
            base_priority = 'high'
        elif frequency >= 3:
            base_priority = 'medium'
        else:
            base_priority = 'low'
        
        # Adjust based on temporal trend
        if temporal_trend == 'increasing':
            if base_priority == 'medium':
                return 'high'
            elif base_priority == 'low':
                return 'medium'
        
        # Adjust based on error type severity
        if any(severe in error_type.lower() for severe in ['conceptual', 'misconception', 'fundamental']):
            if base_priority != 'high':
                return 'high'
        
        return base_priority
    
    def _get_error_description(self, error_type: str, concept: str) -> str:
        """Get human-readable description of error type"""
        descriptions = {
            'unit_error': 'Incorrect unit usage or conversion errors',
            'sign_error': 'Sign convention or directional errors',
            'calculation_error': 'Mathematical calculation mistakes',
            'method_selection_error': 'Incorrect formula or method selection',
            'conceptual_error': 'Fundamental conceptual misunderstanding',
            'velocity_acceleration_confusion': 'Confusion between velocity and acceleration concepts',
            'force_motion_confusion': 'Misunderstanding relationship between force and motion',
            'conservation_error': 'Errors in applying conservation principles'
        }
        
        return descriptions.get(error_type, f'Error pattern in {concept}: {error_type}')
    
    def _get_remediation_strategies(self, error_type: str, concept: str) -> List[str]:
        """Get remediation strategies for specific error types"""
        strategies = {
            'unit_error': [
                'Practice unit conversion exercises',
                'Review dimensional analysis',
                'Use unit checking as verification step'
            ],
            'sign_error': [
                'Establish clear sign conventions',
                'Practice with coordinate system problems',
                'Use vector diagrams for direction checking'
            ],
            'calculation_error': [
                'Double-check mathematical steps',
                'Use estimation for reasonableness checks',
                'Practice algebraic manipulation'
            ],
            'method_selection_error': [
                'Review when to use different formulas',
                'Practice problem categorization',
                'Create decision trees for method selection'
            ],
            'conceptual_error': [
                'Review fundamental concepts',
                'Use conceptual examples and analogies',
                'Practice with concept-focused problems'
            ],
            'velocity_acceleration_confusion': [
                'Compare velocity and acceleration definitions',
                'Use graphical representations',
                'Practice with motion graphs'
            ],
            'force_motion_confusion': [
                'Review Newton\'s laws',
                'Practice free body diagrams',
                'Analyze static vs dynamic situations'
            ]
        }
        
        return strategies.get(error_type, ['Review related concepts', 'Practice similar problems'])
    
    def _calculate_learning_trajectory(self, evidence: List[MasteryEvidence]) -> List[float]:
        """Calculate learning trajectory over time"""
        try:
            if len(evidence) < 3:
                return [e.success for e in evidence]
            
            # Sort evidence by timestamp
            sorted_evidence = sorted(evidence, key=lambda e: e.timestamp)
            
            # Calculate moving average with window size
            window_size = min(5, len(sorted_evidence) // 3)
            trajectory = []
            
            for i in range(window_size, len(sorted_evidence) + 1):
                window_evidence = sorted_evidence[i-window_size:i]
                window_successes = [e.success for e in window_evidence]
                window_weights = [e.weight for e in window_evidence]
                
                weighted_avg = np.average(window_successes, weights=window_weights)
                trajectory.append(weighted_avg)
            
            return trajectory
        
        except Exception as e:
            logger.error(f"❌ Failed to calculate learning trajectory: {e}")
            return []
    
    async def _assess_prerequisite_status(self, user_id: str, concept: str) -> Dict[str, float]:
        """Assess mastery status of prerequisites"""
        prerequisite_status = {}
        
        # This would typically use the concept graph to find prerequisites
        # For now, return a placeholder
        return prerequisite_status
    
    def _generate_next_steps(self, mastery_score: float, error_patterns: List[ErrorPattern], 
                           prerequisite_status: Dict[str, float]) -> List[str]:
        """Generate actionable next steps based on assessment"""
        next_steps = []
        
        if mastery_score >= self.config['mastery_threshold']:
            next_steps.append("Concept mastered - ready for advanced topics")
            next_steps.append("Consider exploring related concepts")
        elif mastery_score >= 0.5:
            next_steps.append("Continue practice to solidify understanding")
            if error_patterns:
                high_priority_errors = [ep for ep in error_patterns if ep.intervention_priority == 'high']
                if high_priority_errors:
                    next_steps.append(f"Focus on addressing {len(high_priority_errors)} high-priority error patterns")
        else:
            next_steps.append("Requires significant additional practice")
            next_steps.append("Consider reviewing prerequisite concepts")
            if error_patterns:
                next_steps.append("Address systematic error patterns before proceeding")
        
        # Add specific recommendations based on error patterns
        for error_pattern in error_patterns[:3]:  # Top 3 error patterns
            if error_pattern.suggested_remediation:
                next_steps.extend(error_pattern.suggested_remediation[:2])
        
        return next_steps[:5]  # Limit to 5 actionable items
    
    def _assess_evidence_quality(self, evidence: List[MasteryEvidence]) -> float:
        """Assess the quality of evidence for mastery assessment"""
        try:
            if not evidence:
                return 0.0
            
            quality_factors = []
            
            # Factor 1: Quantity of evidence
            quantity_score = min(1.0, len(evidence) / 10)  # Optimal around 10 interactions
            quality_factors.append(quantity_score)
            
            # Factor 2: Recency of evidence
            most_recent = max(e.timestamp for e in evidence)
            days_since_recent = (datetime.now() - most_recent).days
            recency_score = max(0.0, 1.0 - days_since_recent / 14)  # Decay over 14 days
            quality_factors.append(recency_score)
            
            # Factor 3: Difficulty distribution
            difficulties = [e.difficulty_level for e in evidence]
            difficulty_std = np.std(difficulties) if len(difficulties) > 1 else 0
            difficulty_score = min(1.0, difficulty_std / 0.5)  # Good if variety in difficulty
            quality_factors.append(difficulty_score)
            
            # Factor 4: Temporal distribution
            timestamps = [e.timestamp for e in evidence]
            if len(timestamps) > 1:
                time_spans = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                             for i in range(len(timestamps) - 1)]
                temporal_consistency = 1.0 - min(1.0, np.std(time_spans) / np.mean(time_spans))
                quality_factors.append(temporal_consistency)
            
            return np.mean(quality_factors)
        
        except Exception as e:
            logger.error(f"❌ Failed to assess evidence quality: {e}")
            return 0.5

    async def detect_misconceptions(self, user_id: str, concept: str) -> List[MisconcepationPattern]:
        """Detect specific misconceptions in student understanding"""
        detected_misconceptions = []
        
        try:
            # Get evidence for analysis
            evidence = await self._collect_mastery_evidence(user_id, concept, 30)
            
            if len(evidence) < 5:
                return detected_misconceptions
            
            # Analyze for known misconception patterns
            for misconception in self.misconception_patterns:
                if concept in misconception.affected_concepts:
                    manifestation_score = await self._calculate_misconception_manifestation(
                        evidence, misconception
                    )
                    
                    if manifestation_score > 0.3:  # Threshold for detection
                        # Create a copy with updated manifestation frequency
                        detected_misconception = MisconcepationPattern(
                            misconception_id=misconception.misconception_id,
                            description=misconception.description,
                            affected_concepts=misconception.affected_concepts,
                            manifestation_frequency=manifestation_score,
                            severity_score=misconception.severity_score,
                            typical_errors=misconception.typical_errors,
                            corrective_strategies=misconception.corrective_strategies
                        )
                        detected_misconceptions.append(detected_misconception)
            
            return detected_misconceptions
        
        except Exception as e:
            logger.error(f"❌ Failed to detect misconceptions: {e}")
            return []
    
    async def _calculate_misconception_manifestation(self, evidence: List[MasteryEvidence], 
                                                   misconception: MisconcepationPattern) -> float:
        """Calculate how strongly a misconception manifests in student evidence"""
        try:
            manifestation_indicators = 0
            total_relevant_evidence = 0
            
            for evidence_item in evidence:
                # Check if this evidence is relevant to the misconception
                if self._is_evidence_relevant_to_misconception(evidence_item, misconception):
                    total_relevant_evidence += 1
                    
                    # Check for manifestation indicators
                    if not evidence_item.success:  # Failed interactions
                        # Check if failure pattern matches misconception
                        if self._failure_matches_misconception(evidence_item, misconception):
                            manifestation_indicators += 1
            
            if total_relevant_evidence == 0:
                return 0.0
            
            return manifestation_indicators / total_relevant_evidence
        
        except Exception as e:
            logger.error(f"❌ Failed to calculate misconception manifestation: {e}")
            return 0.0
    
    def _is_evidence_relevant_to_misconception(self, evidence: MasteryEvidence, 
                                             misconception: MisconcepationPattern) -> bool:
        """Check if evidence is relevant to a specific misconception"""
        # This would analyze the context and interaction type
        # For now, return True for simplicity
        return True
    
    def _failure_matches_misconception(self, evidence: MasteryEvidence, 
                                     misconception: MisconcepationPattern) -> bool:
        """Check if a failure pattern matches a known misconception"""
        # This would analyze the specific failure mode
        # For now, use a simplified approach
        context_str = str(evidence.context).lower()
        
        for typical_error in misconception.typical_errors:
            if any(keyword in context_str for keyword in typical_error.split('_')):
                return True
        
        return False

# Example usage and testing
async def test_concept_mastery_detection():
    """Test function for concept mastery detection"""
    try:
        logger.info("🧪 Testing Concept Mastery Detection System")
        
        # Initialize detector
        detector = ConceptMasteryDetector()
        
        # Test with sample evidence
        sample_evidence = [
            MasteryEvidence(
                interaction_id="test1",
                timestamp=datetime.now() - timedelta(days=1),
                success=True,
                response_time=15.0,
                difficulty_level=1.0,
                context={"problem_type": "basic_kinematics"},
                weight=1.0
            ),
            MasteryEvidence(
                interaction_id="test2", 
                timestamp=datetime.now(),
                success=False,
                response_time=30.0,
                difficulty_level=1.5,
                context={"error_type": "unit_error"},
                weight=1.0
            )
        ]
        
        # Test mastery score calculation
        mastery_score, confidence_interval = detector._calculate_mastery_score(sample_evidence)
        logger.info(f"✅ Sample mastery score: {mastery_score:.2f}, CI: {confidence_interval}")
        
        # Test learning trajectory
        trajectory = detector._calculate_learning_trajectory(sample_evidence)
        logger.info(f"✅ Learning trajectory: {trajectory}")
        
        # Test error classification
        error_types = detector._extract_error_types(
            "Unit conversion error in velocity calculation",
            {"error_type": "unit_error"},
            None, None, "kinematics"
        )
        logger.info(f"✅ Error types: {error_types}")
        
        logger.info("✅ Concept Mastery Detection test completed")
        
    except Exception as e:
        logger.error(f"❌ Concept Mastery Detection test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_concept_mastery_detection())