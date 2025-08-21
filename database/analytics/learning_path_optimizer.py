#!/usr/bin/env python3
"""
Learning Path Recommendation Engine for Physics Assistant
Graph-based optimization system for generating personalized learning paths,
optimizing difficulty progression, and recommending next concepts based on
student knowledge state and learning objectives.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import networkx as nx
from heapq import heappush, heappop
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LearningObjective:
    """Learning objective with constraints and preferences"""
    target_concepts: List[str]
    time_constraint: Optional[float] = None  # hours
    difficulty_preference: str = "adaptive"  # adaptive, easy, moderate, challenging
    learning_style: str = "mixed"  # visual, analytical, practical, mixed
    priority_level: str = "medium"  # low, medium, high
    completion_deadline: Optional[datetime] = None

@dataclass
class ConceptNode:
    """Enhanced concept node with learning analytics"""
    name: str
    category: str
    difficulty: float
    estimated_time: float  # hours to master
    prerequisites: List[str]
    successors: List[str]
    mastery_evidence_required: int
    learning_resources: List[str] = field(default_factory=list)
    common_misconceptions: List[str] = field(default_factory=list)

@dataclass
class LearningPath:
    """Optimized learning path with detailed metrics"""
    path_id: str
    student_id: str
    concept_sequence: List[str]
    estimated_total_time: float
    difficulty_progression: List[float]
    success_probability: float
    adaptive_checkpoints: List[int]  # Indices for adaptation points
    alternative_paths: List[List[str]] = field(default_factory=list)
    personalization_factors: Dict[str, float] = field(default_factory=dict)
    creation_timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class PathSegment:
    """Individual segment of a learning path"""
    from_concept: str
    to_concept: str
    transition_difficulty: float
    estimated_time: float
    success_rate: float
    required_mastery_level: float
    adaptive_hints: List[str] = field(default_factory=list)

@dataclass
class StudentState:
    """Current student knowledge and learning state"""
    concept_masteries: Dict[str, float]
    learning_velocity: float
    engagement_level: float
    preferred_difficulty: float
    strong_areas: List[str]
    weak_areas: List[str]
    learning_patterns: Dict[str, Any]

class LearningPathOptimizer:
    """Advanced learning path recommendation and optimization engine"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.concept_graph = None
        self.transition_matrix = {}
        self.success_rate_matrix = {}
        self.student_models = {}
        
        # Configuration parameters
        self.config = {
            'mastery_threshold': 0.75,
            'max_path_length': 10,
            'difficulty_step_limit': 0.3,
            'success_rate_threshold': 0.6,
            'time_weight': 0.3,
            'difficulty_weight': 0.4,
            'success_weight': 0.3,
            'personalization_weight': 0.5,
            'exploration_factor': 0.1
        }
        
        # Initialize optimization algorithms
        self.path_algorithms = {
            'shortest_time': self._shortest_time_path,
            'highest_success': self._highest_success_path,
            'balanced_optimization': self._balanced_optimization_path,
            'adaptive_difficulty': self._adaptive_difficulty_path,
            'personalized_optimal': self._personalized_optimal_path
        }
    
    async def initialize(self):
        """Initialize the learning path optimizer"""
        try:
            logger.info("🚀 Initializing Learning Path Optimizer")
            
            # Load enhanced concept graph
            await self._load_enhanced_concept_graph()
            
            # Build transition matrices
            await self._build_transition_matrices()
            
            # Load student models
            await self._load_student_models()
            
            # Initialize success rate predictions
            await self._initialize_success_predictors()
            
            logger.info("✅ Learning Path Optimizer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Learning Path Optimizer: {e}")
            return False
    
    async def _load_enhanced_concept_graph(self):
        """Load enhanced concept graph with learning analytics"""
        if not self.db_manager:
            self._create_sample_concept_graph()
            return
        
        try:
            # Get concepts with enhanced metadata
            concepts_query = """
            MATCH (c:Concept)
            RETURN c.name as name, c.category as category, 
                   c.difficulty as difficulty, c.estimated_time as estimated_time,
                   c.description as description, c.learning_resources as learning_resources
            """
            
            # Get prerequisite relationships
            prereq_query = """
            MATCH (prereq:Concept)-[:PREREQUISITE]->(concept:Concept)
            RETURN prereq.name as prerequisite, concept.name as concept
            """
            
            # Get successor relationships
            successor_query = """
            MATCH (concept:Concept)-[:LEADS_TO]->(successor:Concept)
            RETURN concept.name as concept, successor.name as successor
            """
            
            concepts = await self.db_manager.neo4j.run_query(concepts_query)
            prerequisites = await self.db_manager.neo4j.run_query(prereq_query)
            successors = await self.db_manager.neo4j.run_query(successor_query)
            
            # Build enhanced concept graph
            self.concept_graph = nx.DiGraph()
            
            # Add concept nodes with enhanced data
            for concept in concepts:
                concept_node = ConceptNode(
                    name=concept['name'],
                    category=concept['category'],
                    difficulty=concept.get('difficulty', 1.0),
                    estimated_time=concept.get('estimated_time', 2.0),
                    prerequisites=[],
                    successors=[],
                    mastery_evidence_required=5,
                    learning_resources=concept.get('learning_resources', [])
                )
                
                self.concept_graph.add_node(concept['name'], concept_data=concept_node)
            
            # Add prerequisite edges
            for prereq in prerequisites:
                if self.concept_graph.has_node(prereq['prerequisite']) and self.concept_graph.has_node(prereq['concept']):
                    self.concept_graph.add_edge(
                        prereq['prerequisite'],
                        prereq['concept'],
                        relationship_type='prerequisite',
                        weight=1.0
                    )
                    
                    # Update concept node data
                    concept_data = self.concept_graph.nodes[prereq['concept']]['concept_data']
                    concept_data.prerequisites.append(prereq['prerequisite'])
            
            # Add successor edges
            for succ in successors:
                if self.concept_graph.has_node(succ['concept']) and self.concept_graph.has_node(succ['successor']):
                    if not self.concept_graph.has_edge(succ['concept'], succ['successor']):
                        self.concept_graph.add_edge(
                            succ['concept'],
                            succ['successor'],
                            relationship_type='leads_to',
                            weight=1.0
                        )
                    
                    # Update concept node data
                    concept_data = self.concept_graph.nodes[succ['concept']]['concept_data']
                    concept_data.successors.append(succ['successor'])
            
            logger.info(f"📊 Enhanced concept graph loaded: {len(concepts)} concepts")
            
        except Exception as e:
            logger.error(f"❌ Failed to load enhanced concept graph: {e}")
            self._create_sample_concept_graph()
    
    def _create_sample_concept_graph(self):
        """Create sample concept graph for testing"""
        self.concept_graph = nx.DiGraph()
        
        sample_concepts = [
            ConceptNode("basic_math", "foundation", 0.3, 1.0, [], ["kinematics_1d"], 3),
            ConceptNode("kinematics_1d", "mechanics", 0.5, 2.0, ["basic_math"], ["kinematics_2d"], 5),
            ConceptNode("kinematics_2d", "mechanics", 0.7, 3.0, ["kinematics_1d"], ["forces"], 5),
            ConceptNode("forces", "mechanics", 0.8, 2.5, ["kinematics_2d"], ["energy"], 6),
            ConceptNode("energy", "mechanics", 0.9, 3.0, ["forces"], ["momentum"], 6),
            ConceptNode("momentum", "mechanics", 0.8, 2.5, ["forces"], ["angular_motion"], 5),
            ConceptNode("angular_motion", "mechanics", 1.0, 4.0, ["momentum"], [], 7)
        ]
        
        for concept in sample_concepts:
            self.concept_graph.add_node(concept.name, concept_data=concept)
        
        # Add edges based on prerequisites
        for concept in sample_concepts:
            for prereq in concept.prerequisites:
                if self.concept_graph.has_node(prereq):
                    self.concept_graph.add_edge(prereq, concept.name, relationship_type='prerequisite')
    
    async def _build_transition_matrices(self):
        """Build transition matrices for path optimization"""
        try:
            if not self.concept_graph:
                return
            
            concepts = list(self.concept_graph.nodes())
            n_concepts = len(concepts)
            
            # Initialize matrices
            self.transition_matrix = np.zeros((n_concepts, n_concepts))
            self.success_rate_matrix = np.ones((n_concepts, n_concepts)) * 0.5
            
            concept_to_idx = {concept: i for i, concept in enumerate(concepts)}
            
            # Build transition matrix based on graph structure
            for i, from_concept in enumerate(concepts):
                from_data = self.concept_graph.nodes[from_concept]['concept_data']
                
                for j, to_concept in enumerate(concepts):
                    if i == j:
                        continue
                    
                    to_data = self.concept_graph.nodes[to_concept]['concept_data']
                    
                    # Calculate transition weight
                    if self.concept_graph.has_edge(from_concept, to_concept):
                        # Direct connection - high weight
                        difficulty_diff = abs(to_data.difficulty - from_data.difficulty)
                        if difficulty_diff <= self.config['difficulty_step_limit']:
                            weight = 1.0 - difficulty_diff
                        else:
                            weight = 0.1  # Penalize large difficulty jumps
                    else:
                        # No direct connection - calculate based on similarity and prerequisites
                        if from_concept in to_data.prerequisites:
                            weight = 0.8  # Good transition
                        elif to_concept in from_data.successors:
                            weight = 0.9  # Excellent transition
                        else:
                            # Check category similarity
                            if from_data.category == to_data.category:
                                difficulty_diff = abs(to_data.difficulty - from_data.difficulty)
                                weight = max(0.1, 0.5 - difficulty_diff)
                            else:
                                weight = 0.1
                    
                    self.transition_matrix[i, j] = weight
            
            # Initialize success rate matrix with historical data if available
            if self.db_manager:
                await self._populate_success_rates(concept_to_idx)
            
            logger.info(f"📊 Transition matrices built for {n_concepts} concepts")
            
        except Exception as e:
            logger.error(f"❌ Failed to build transition matrices: {e}")
    
    async def _populate_success_rates(self, concept_to_idx: Dict[str, int]):
        """Populate success rate matrix with historical data"""
        try:
            async with self.db_manager.postgres.get_connection() as conn:
                # Get transition success rates from historical data
                transition_stats = await conn.fetch("""
                    WITH concept_sequences AS (
                        SELECT user_id, agent_type as concept, created_at, success,
                               LAG(agent_type) OVER (PARTITION BY user_id ORDER BY created_at) as prev_concept
                        FROM interactions 
                        WHERE agent_type IS NOT NULL AND created_at >= $1
                    )
                    SELECT prev_concept, concept, 
                           COUNT(*) as total_transitions,
                           SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_transitions
                    FROM concept_sequences 
                    WHERE prev_concept IS NOT NULL
                    GROUP BY prev_concept, concept
                    HAVING COUNT(*) >= 3
                """, datetime.now() - timedelta(days=30))
                
                # Update success rate matrix
                for stat in transition_stats:
                    from_concept = stat['prev_concept']
                    to_concept = stat['concept']
                    
                    if from_concept in concept_to_idx and to_concept in concept_to_idx:
                        from_idx = concept_to_idx[from_concept]
                        to_idx = concept_to_idx[to_concept]
                        
                        success_rate = stat['successful_transitions'] / stat['total_transitions']
                        self.success_rate_matrix[from_idx, to_idx] = success_rate
            
            logger.info("📊 Success rate matrix populated with historical data")
            
        except Exception as e:
            logger.error(f"❌ Failed to populate success rates: {e}")
    
    async def _load_student_models(self):
        """Load student learning models"""
        self.student_models = {}
        
        if not self.db_manager:
            return
        
        try:
            async with self.db_manager.postgres.get_connection() as conn:
                students = await conn.fetch("""
                    SELECT u.id, u.username,
                           AVG(CASE WHEN i.success THEN 1.0 ELSE 0.0 END) as avg_success_rate,
                           COUNT(i.id) as total_interactions
                    FROM users u
                    LEFT JOIN interactions i ON u.id = i.user_id 
                    WHERE u.is_active = TRUE AND i.created_at >= $1
                    GROUP BY u.id, u.username
                    HAVING COUNT(i.id) >= 5
                """, datetime.now() - timedelta(days=30))
                
                for student in students:
                    student_id = str(student['id'])
                    self.student_models[student_id] = {
                        'avg_success_rate': student['avg_success_rate'],
                        'total_interactions': student['total_interactions'],
                        'learning_velocity': student['avg_success_rate'] * 0.5,  # Simplified
                        'preferred_difficulty': 0.5 + student['avg_success_rate'] * 0.3
                    }
            
            logger.info(f"👥 Loaded models for {len(self.student_models)} students")
            
        except Exception as e:
            logger.error(f"❌ Failed to load student models: {e}")
    
    async def _initialize_success_predictors(self):
        """Initialize machine learning models for success prediction"""
        # Placeholder for ML model initialization
        # In a full implementation, this would load trained models
        pass
    
    async def generate_learning_path(self, student_id: str, objective: LearningObjective,
                                   algorithm: str = "personalized_optimal") -> LearningPath:
        """Generate optimized learning path for student objective"""
        try:
            logger.info(f"🎯 Generating learning path for student {student_id}")
            
            # Get current student state
            student_state = await self._get_student_state(student_id)
            
            # Determine starting concepts
            starting_concepts = await self._identify_starting_concepts(student_state, objective)
            
            # Select algorithm
            path_algorithm = self.path_algorithms.get(algorithm, self._personalized_optimal_path)
            
            # Generate path
            path = await path_algorithm(student_state, objective, starting_concepts)
            
            # Optimize path
            optimized_path = await self._optimize_path(path, student_state, objective)
            
            # Add adaptive checkpoints
            optimized_path.adaptive_checkpoints = self._identify_adaptive_checkpoints(optimized_path)
            
            # Generate alternative paths
            optimized_path.alternative_paths = await self._generate_alternative_paths(
                student_state, objective, optimized_path.concept_sequence
            )
            
            logger.info(f"✅ Generated learning path with {len(optimized_path.concept_sequence)} concepts")
            return optimized_path
            
        except Exception as e:
            logger.error(f"❌ Failed to generate learning path: {e}")
            return LearningPath(
                path_id=f"error_{student_id}",
                student_id=student_id,
                concept_sequence=[],
                estimated_total_time=0.0,
                difficulty_progression=[],
                success_probability=0.0,
                adaptive_checkpoints=[]
            )
    
    async def _get_student_state(self, student_id: str) -> StudentState:
        """Get current student knowledge and learning state"""
        try:
            # Get concept masteries from database
            concept_masteries = {}
            
            if self.db_manager:
                async with self.db_manager.postgres.get_connection() as conn:
                    progress_data = await conn.fetch("""
                        SELECT topic, proficiency_score / 100.0 as mastery_level
                        FROM user_progress 
                        WHERE user_id = $1
                    """, student_id)
                    
                    for progress in progress_data:
                        concept_masteries[progress['topic']] = progress['mastery_level']
            
            # Get learning patterns from student model
            student_model = self.student_models.get(student_id, {})
            
            # Categorize strong and weak areas
            strong_areas = [concept for concept, mastery in concept_masteries.items() 
                          if mastery >= 0.75]
            weak_areas = [concept for concept, mastery in concept_masteries.items() 
                         if mastery < 0.5]
            
            return StudentState(
                concept_masteries=concept_masteries,
                learning_velocity=student_model.get('learning_velocity', 0.5),
                engagement_level=student_model.get('avg_success_rate', 0.5),
                preferred_difficulty=student_model.get('preferred_difficulty', 0.5),
                strong_areas=strong_areas,
                weak_areas=weak_areas,
                learning_patterns=student_model
            )
        
        except Exception as e:
            logger.error(f"❌ Failed to get student state: {e}")
            return StudentState(
                concept_masteries={},
                learning_velocity=0.5,
                engagement_level=0.5,
                preferred_difficulty=0.5,
                strong_areas=[],
                weak_areas=[],
                learning_patterns={}
            )
    
    async def _identify_starting_concepts(self, student_state: StudentState, 
                                        objective: LearningObjective) -> List[str]:
        """Identify optimal starting concepts for the learning path"""
        starting_concepts = []
        
        if not self.concept_graph:
            return objective.target_concepts[:1]  # Fallback to first target
        
        try:
            for target in objective.target_concepts:
                if target not in self.concept_graph:
                    continue
                
                # Check if student already masters this concept
                current_mastery = student_state.concept_masteries.get(target, 0.0)
                if current_mastery >= self.config['mastery_threshold']:
                    continue  # Skip already mastered concepts
                
                # Find the best starting point for this target
                best_start = self._find_optimal_starting_point(target, student_state)
                if best_start and best_start not in starting_concepts:
                    starting_concepts.append(best_start)
            
            # If no specific starting points found, use prerequisites of targets
            if not starting_concepts:
                for target in objective.target_concepts:
                    if target in self.concept_graph:
                        concept_data = self.concept_graph.nodes[target]['concept_data']
                        for prereq in concept_data.prerequisites:
                            if prereq not in starting_concepts:
                                prereq_mastery = student_state.concept_masteries.get(prereq, 0.0)
                                if prereq_mastery < self.config['mastery_threshold']:
                                    starting_concepts.append(prereq)
            
            return starting_concepts if starting_concepts else objective.target_concepts
        
        except Exception as e:
            logger.error(f"❌ Failed to identify starting concepts: {e}")
            return objective.target_concepts[:1]
    
    def _find_optimal_starting_point(self, target_concept: str, student_state: StudentState) -> str:
        """Find optimal starting point for reaching a target concept"""
        if target_concept not in self.concept_graph:
            return target_concept
        
        # Use BFS to find prerequisites the student hasn't mastered
        queue = deque([target_concept])
        visited = set()
        unmastered_prerequisites = []
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            
            if current in self.concept_graph:
                concept_data = self.concept_graph.nodes[current]['concept_data']
                
                for prereq in concept_data.prerequisites:
                    mastery = student_state.concept_masteries.get(prereq, 0.0)
                    if mastery < self.config['mastery_threshold']:
                        unmastered_prerequisites.append((prereq, mastery))
                        queue.append(prereq)
        
        # Return the prerequisite with lowest mastery (needs most work)
        if unmastered_prerequisites:
            unmastered_prerequisites.sort(key=lambda x: x[1])
            return unmastered_prerequisites[0][0]
        
        return target_concept
    
    async def _personalized_optimal_path(self, student_state: StudentState, 
                                       objective: LearningObjective, 
                                       starting_concepts: List[str]) -> LearningPath:
        """Generate personalized optimal learning path using A* algorithm"""
        try:
            path_segments = []
            total_time = 0.0
            difficulty_progression = []
            
            # Use modified A* for path finding
            for i, start_concept in enumerate(starting_concepts):
                if i < len(objective.target_concepts):
                    target = objective.target_concepts[i]
                    
                    # Find path from start to target
                    segment_path = await self._astar_learning_path(
                        start_concept, target, student_state, objective
                    )
                    
                    if segment_path:
                        path_segments.extend(segment_path)
                        
                        # Calculate metrics for this segment
                        for concept in segment_path:
                            if concept in self.concept_graph:
                                concept_data = self.concept_graph.nodes[concept]['concept_data']
                                total_time += concept_data.estimated_time
                                difficulty_progression.append(concept_data.difficulty)
            
            # Remove duplicates while preserving order
            unique_path = []
            seen = set()
            for concept in path_segments:
                if concept not in seen:
                    unique_path.append(concept)
                    seen.add(concept)
            
            # Calculate success probability
            success_probability = self._calculate_path_success_probability(
                unique_path, student_state
            )
            
            # Create personalization factors
            personalization_factors = {
                'difficulty_adjustment': self._calculate_difficulty_adjustment(student_state),
                'time_preference': objective.time_constraint or 10.0,
                'success_weight': student_state.engagement_level,
                'learning_velocity_factor': student_state.learning_velocity
            }
            
            return LearningPath(
                path_id=f"optimal_{student_state.concept_masteries.get('user_id', 'unknown')}_{datetime.now().timestamp()}",
                student_id=student_state.concept_masteries.get('user_id', 'unknown'),
                concept_sequence=unique_path,
                estimated_total_time=total_time,
                difficulty_progression=difficulty_progression,
                success_probability=success_probability,
                adaptive_checkpoints=[],
                personalization_factors=personalization_factors
            )
        
        except Exception as e:
            logger.error(f"❌ Failed to generate personalized optimal path: {e}")
            return LearningPath(
                path_id="error_path",
                student_id="unknown",
                concept_sequence=[],
                estimated_total_time=0.0,
                difficulty_progression=[],
                success_probability=0.0,
                adaptive_checkpoints=[]
            )
    
    async def _astar_learning_path(self, start: str, goal: str, student_state: StudentState,
                                 objective: LearningObjective) -> List[str]:
        """A* algorithm adapted for learning path optimization"""
        try:
            if not self.concept_graph or start not in self.concept_graph or goal not in self.concept_graph:
                return [start, goal] if start != goal else [start]
            
            # Priority queue: (f_score, g_score, current_node, path)
            open_set = [(0, 0, start, [start])]
            closed_set = set()
            
            while open_set:
                f_score, g_score, current, path = heappop(open_set)
                
                if current == goal:
                    return path
                
                if current in closed_set:
                    continue
                
                closed_set.add(current)
                
                # Explore neighbors
                if current in self.concept_graph:
                    for neighbor in self.concept_graph.successors(current):
                        if neighbor in closed_set:
                            continue
                        
                        # Calculate costs
                        transition_cost = self._calculate_transition_cost(
                            current, neighbor, student_state, objective
                        )
                        
                        tentative_g_score = g_score + transition_cost
                        h_score = self._calculate_heuristic(neighbor, goal, student_state)
                        f_score = tentative_g_score + h_score
                        
                        new_path = path + [neighbor]
                        
                        # Avoid cycles and overly long paths
                        if len(new_path) <= self.config['max_path_length']:
                            heappush(open_set, (f_score, tentative_g_score, neighbor, new_path))
            
            # If no path found, return direct connection
            return [start, goal] if start != goal else [start]
        
        except Exception as e:
            logger.error(f"❌ A* learning path failed: {e}")
            return [start, goal] if start != goal else [start]
    
    def _calculate_transition_cost(self, from_concept: str, to_concept: str,
                                 student_state: StudentState, objective: LearningObjective) -> float:
        """Calculate cost of transitioning between concepts"""
        try:
            if not self.concept_graph:
                return 1.0
            
            from_data = self.concept_graph.nodes[from_concept]['concept_data']
            to_data = self.concept_graph.nodes[to_concept]['concept_data']
            
            # Base cost factors
            time_cost = to_data.estimated_time * self.config['time_weight']
            difficulty_cost = abs(to_data.difficulty - from_data.difficulty) * self.config['difficulty_weight']
            
            # Success probability cost
            success_prob = self._estimate_transition_success(from_concept, to_concept, student_state)
            success_cost = (1.0 - success_prob) * self.config['success_weight']
            
            # Personalization adjustments
            current_mastery = student_state.concept_masteries.get(to_concept, 0.0)
            mastery_adjustment = max(0.1, 1.0 - current_mastery)  # Less cost if already partially known
            
            # Difficulty preference adjustment
            difficulty_preference_cost = 0.0
            if objective.difficulty_preference == "easy" and to_data.difficulty > 0.7:
                difficulty_preference_cost = 0.5
            elif objective.difficulty_preference == "challenging" and to_data.difficulty < 0.5:
                difficulty_preference_cost = 0.3
            
            total_cost = (time_cost + difficulty_cost + success_cost) * mastery_adjustment + difficulty_preference_cost
            
            return max(0.1, total_cost)
        
        except Exception as e:
            logger.error(f"❌ Failed to calculate transition cost: {e}")
            return 1.0
    
    def _calculate_heuristic(self, current: str, goal: str, student_state: StudentState) -> float:
        """Calculate heuristic for A* algorithm"""
        try:
            if not self.concept_graph or current == goal:
                return 0.0
            
            # Try to find shortest path in concept graph
            try:
                shortest_path_length = nx.shortest_path_length(self.concept_graph, current, goal)
                base_heuristic = shortest_path_length
            except nx.NetworkXNoPath:
                # No direct path, estimate based on concept similarity
                current_data = self.concept_graph.nodes[current]['concept_data']
                goal_data = self.concept_graph.nodes[goal]['concept_data']
                
                # Category similarity
                category_similarity = 1.0 if current_data.category == goal_data.category else 2.0
                
                # Difficulty similarity
                difficulty_diff = abs(goal_data.difficulty - current_data.difficulty)
                
                base_heuristic = category_similarity + difficulty_diff
            
            # Adjust heuristic based on student state
            current_mastery = student_state.concept_masteries.get(current, 0.0)
            goal_mastery = student_state.concept_masteries.get(goal, 0.0)
            
            # Lower heuristic if student is closer to mastering concepts on the path
            mastery_factor = 1.0 - (current_mastery + goal_mastery) / 2.0
            
            return base_heuristic * mastery_factor
        
        except Exception as e:
            logger.error(f"❌ Failed to calculate heuristic: {e}")
            return 1.0
    
    def _estimate_transition_success(self, from_concept: str, to_concept: str,
                                   student_state: StudentState) -> float:
        """Estimate success probability for concept transition"""
        try:
            # Base success rate from historical data
            base_success = 0.6  # Default
            
            if hasattr(self, 'success_rate_matrix') and self.concept_graph:
                concepts = list(self.concept_graph.nodes())
                if from_concept in concepts and to_concept in concepts:
                    from_idx = concepts.index(from_concept)
                    to_idx = concepts.index(to_concept)
                    base_success = self.success_rate_matrix[from_idx, to_idx]
            
            # Adjust for student's current mastery
            from_mastery = student_state.concept_masteries.get(from_concept, 0.0)
            to_mastery = student_state.concept_masteries.get(to_concept, 0.0)
            
            # Higher from_mastery increases success probability
            mastery_bonus = from_mastery * 0.3
            
            # Partial to_mastery also helps
            partial_mastery_bonus = to_mastery * 0.2
            
            # Adjust for student's general performance
            performance_factor = student_state.engagement_level
            
            estimated_success = base_success + mastery_bonus + partial_mastery_bonus
            estimated_success *= performance_factor
            
            return max(0.1, min(0.95, estimated_success))
        
        except Exception as e:
            logger.error(f"❌ Failed to estimate transition success: {e}")
            return 0.6
    
    def _calculate_path_success_probability(self, path: List[str], student_state: StudentState) -> float:
        """Calculate overall success probability for entire path"""
        if not path:
            return 0.0
        
        if len(path) == 1:
            return student_state.concept_masteries.get(path[0], 0.0)
        
        # Calculate cumulative success probability
        cumulative_success = 1.0
        
        for i in range(len(path) - 1):
            transition_success = self._estimate_transition_success(
                path[i], path[i + 1], student_state
            )
            cumulative_success *= transition_success
        
        return cumulative_success
    
    def _calculate_difficulty_adjustment(self, student_state: StudentState) -> float:
        """Calculate difficulty adjustment factor for student"""
        # Base adjustment on student's success rate and preferred difficulty
        base_adjustment = student_state.preferred_difficulty
        
        # Adjust based on recent performance
        if student_state.engagement_level > 0.8:
            return min(1.0, base_adjustment + 0.2)  # Increase difficulty
        elif student_state.engagement_level < 0.5:
            return max(0.3, base_adjustment - 0.2)  # Decrease difficulty
        
        return base_adjustment
    
    async def _optimize_path(self, path: LearningPath, student_state: StudentState,
                           objective: LearningObjective) -> LearningPath:
        """Optimize learning path using advanced techniques"""
        try:
            # Optimize sequence order
            optimized_sequence = await self._optimize_sequence_order(
                path.concept_sequence, student_state, objective
            )
            
            # Recalculate metrics
            total_time = 0.0
            difficulty_progression = []
            
            if self.concept_graph:
                for concept in optimized_sequence:
                    if concept in self.concept_graph:
                        concept_data = self.concept_graph.nodes[concept]['concept_data']
                        total_time += concept_data.estimated_time
                        difficulty_progression.append(concept_data.difficulty)
            
            # Recalculate success probability
            success_probability = self._calculate_path_success_probability(
                optimized_sequence, student_state
            )
            
            # Update path
            path.concept_sequence = optimized_sequence
            path.estimated_total_time = total_time
            path.difficulty_progression = difficulty_progression
            path.success_probability = success_probability
            
            return path
        
        except Exception as e:
            logger.error(f"❌ Failed to optimize path: {e}")
            return path
    
    async def _optimize_sequence_order(self, sequence: List[str], student_state: StudentState,
                                     objective: LearningObjective) -> List[str]:
        """Optimize the order of concepts in the sequence"""
        try:
            if len(sequence) <= 2:
                return sequence
            
            # Use topological sort with custom ordering
            if self.concept_graph:
                # Create subgraph with only concepts in the sequence
                subgraph = self.concept_graph.subgraph(sequence)
                
                # Topological sort respecting prerequisites
                try:
                    topo_order = list(nx.topological_sort(subgraph))
                    
                    # Filter to maintain only concepts in original sequence
                    optimized_order = [concept for concept in topo_order if concept in sequence]
                    
                    # Add any missing concepts at the end
                    for concept in sequence:
                        if concept not in optimized_order:
                            optimized_order.append(concept)
                    
                    return optimized_order
                
                except nx.NetworkXError:
                    # Graph has cycles, use custom ordering
                    return self._custom_sequence_optimization(sequence, student_state, objective)
            
            return sequence
        
        except Exception as e:
            logger.error(f"❌ Failed to optimize sequence order: {e}")
            return sequence
    
    def _custom_sequence_optimization(self, sequence: List[str], student_state: StudentState,
                                    objective: LearningObjective) -> List[str]:
        """Custom sequence optimization when topological sort fails"""
        try:
            # Sort by difficulty and mastery level
            concept_scores = []
            
            for concept in sequence:
                current_mastery = student_state.concept_masteries.get(concept, 0.0)
                
                if self.concept_graph and concept in self.concept_graph:
                    concept_data = self.concept_graph.nodes[concept]['concept_data']
                    difficulty = concept_data.difficulty
                else:
                    difficulty = 0.5
                
                # Score prioritizes lower difficulty and higher current mastery
                score = (1.0 - difficulty) + current_mastery
                concept_scores.append((score, concept))
            
            # Sort by score (higher score = should come first)
            concept_scores.sort(reverse=True)
            
            return [concept for score, concept in concept_scores]
        
        except Exception as e:
            logger.error(f"❌ Custom sequence optimization failed: {e}")
            return sequence
    
    def _identify_adaptive_checkpoints(self, path: LearningPath) -> List[int]:
        """Identify points in the path where adaptation should occur"""
        checkpoints = []
        
        try:
            sequence = path.concept_sequence
            difficulty_progression = path.difficulty_progression
            
            if len(sequence) <= 2:
                return checkpoints
            
            # Add checkpoint at start
            checkpoints.append(0)
            
            # Add checkpoints at significant difficulty increases
            for i in range(1, len(difficulty_progression)):
                if i > 0:
                    difficulty_increase = difficulty_progression[i] - difficulty_progression[i-1]
                    if difficulty_increase > 0.3:  # Significant jump
                        checkpoints.append(i)
            
            # Add checkpoint at midpoint for long paths
            if len(sequence) > 6:
                mid_point = len(sequence) // 2
                if mid_point not in checkpoints:
                    checkpoints.append(mid_point)
            
            # Add checkpoint at end
            if len(sequence) - 1 not in checkpoints:
                checkpoints.append(len(sequence) - 1)
            
            return sorted(checkpoints)
        
        except Exception as e:
            logger.error(f"❌ Failed to identify adaptive checkpoints: {e}")
            return [0]
    
    async def _generate_alternative_paths(self, student_state: StudentState,
                                        objective: LearningObjective,
                                        main_path: List[str]) -> List[List[str]]:
        """Generate alternative learning paths"""
        alternatives = []
        
        try:
            # Generate path with different optimization criteria
            for algorithm in ['shortest_time', 'highest_success', 'balanced_optimization']:
                if algorithm != 'personalized_optimal':
                    alt_objective = LearningObjective(
                        target_concepts=objective.target_concepts,
                        difficulty_preference='easy' if algorithm == 'highest_success' else objective.difficulty_preference
                    )
                    
                    starting_concepts = await self._identify_starting_concepts(student_state, alt_objective)
                    alt_path_algorithm = self.path_algorithms.get(algorithm, self._balanced_optimization_path)
                    
                    alt_path = await alt_path_algorithm(student_state, alt_objective, starting_concepts)
                    
                    if alt_path.concept_sequence and alt_path.concept_sequence != main_path:
                        alternatives.append(alt_path.concept_sequence)
            
            return alternatives[:3]  # Limit to 3 alternatives
        
        except Exception as e:
            logger.error(f"❌ Failed to generate alternative paths: {e}")
            return []
    
    # Additional path algorithms
    async def _shortest_time_path(self, student_state: StudentState, objective: LearningObjective,
                                starting_concepts: List[str]) -> LearningPath:
        """Generate path optimized for minimal time"""
        # Simplified implementation - prioritize concepts with lowest estimated time
        path_sequence = []
        
        if self.concept_graph:
            for start in starting_concepts:
                if start in self.concept_graph:
                    concept_data = self.concept_graph.nodes[start]['concept_data']
                    path_sequence.append(start)
        
        return LearningPath(
            path_id=f"shortest_time_{len(path_sequence)}",
            student_id="unknown",
            concept_sequence=path_sequence,
            estimated_total_time=sum(2.0 for _ in path_sequence),  # Simplified
            difficulty_progression=[0.5] * len(path_sequence),
            success_probability=0.7,
            adaptive_checkpoints=[]
        )
    
    async def _highest_success_path(self, student_state: StudentState, objective: LearningObjective,
                                  starting_concepts: List[str]) -> LearningPath:
        """Generate path optimized for highest success probability"""
        # Focus on concepts where student has highest chance of success
        return await self._personalized_optimal_path(student_state, objective, starting_concepts)
    
    async def _balanced_optimization_path(self, student_state: StudentState, objective: LearningObjective,
                                        starting_concepts: List[str]) -> LearningPath:
        """Generate balanced path considering multiple factors equally"""
        return await self._personalized_optimal_path(student_state, objective, starting_concepts)
    
    async def _adaptive_difficulty_path(self, student_state: StudentState, objective: LearningObjective,
                                      starting_concepts: List[str]) -> LearningPath:
        """Generate path with adaptive difficulty progression"""
        return await self._personalized_optimal_path(student_state, objective, starting_concepts)

# Example usage and testing
async def test_learning_path_optimizer():
    """Test function for learning path optimizer"""
    try:
        logger.info("🧪 Testing Learning Path Optimizer")
        
        # Initialize optimizer
        optimizer = LearningPathOptimizer()
        await optimizer.initialize()
        
        # Create test student state
        test_student_state = StudentState(
            concept_masteries={"basic_math": 0.8, "kinematics_1d": 0.6},
            learning_velocity=0.5,
            engagement_level=0.7,
            preferred_difficulty=0.6,
            strong_areas=["basic_math"],
            weak_areas=["forces"],
            learning_patterns={}
        )
        
        # Create test objective
        test_objective = LearningObjective(
            target_concepts=["forces", "energy"],
            difficulty_preference="adaptive",
            learning_style="mixed"
        )
        
        # Generate learning path
        path = await optimizer.generate_learning_path("test_student", test_objective)
        
        logger.info(f"✅ Generated path: {path.concept_sequence}")
        logger.info(f"✅ Estimated time: {path.estimated_total_time:.1f} hours")
        logger.info(f"✅ Success probability: {path.success_probability:.2f}")
        logger.info(f"✅ Adaptive checkpoints: {path.adaptive_checkpoints}")
        
        logger.info("✅ Learning Path Optimizer test completed")
        
    except Exception as e:
        logger.error(f"❌ Learning Path Optimizer test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_learning_path_optimizer())