#!/usr/bin/env python3
"""
Personalized Recommendation System for Physics Assistant Phase 6
Implements collaborative filtering, content-based recommendations,
and hybrid approaches for personalized learning experiences.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from collections import defaultdict, Counter
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

class RecommendationType(Enum):
    CONTENT = "content"
    LEARNING_PATH = "learning_path"
    STUDY_SCHEDULE = "study_schedule"
    PEER_GROUP = "peer_group"
    RESOURCE = "resource"
    DIFFICULTY_ADJUSTMENT = "difficulty_adjustment"

class RecommendationReason(Enum):
    COLLABORATIVE_FILTERING = "collaborative_filtering"
    CONTENT_BASED = "content_based"
    KNOWLEDGE_GAP = "knowledge_gap"
    LEARNING_STYLE = "learning_style"
    PERFORMANCE_BASED = "performance_based"
    PEER_SUCCESS = "peer_success"
    TEMPORAL_PATTERN = "temporal_pattern"

@dataclass
class LearningResource:
    """Learning resource with metadata"""
    resource_id: str
    title: str
    content_type: str          # "video", "text", "interactive", "problem", "simulation"
    topic: str
    difficulty_level: float    # 0-1 scale
    duration_minutes: int
    learning_objectives: List[str]
    prerequisites: List[str]
    tags: List[str]
    quality_score: float       # Based on student feedback and outcomes
    engagement_score: float    # How engaging students find it
    effectiveness_score: float # Learning outcome improvement
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StudySession:
    """Study session recommendation"""
    session_id: str
    student_id: str
    recommended_duration: int  # minutes
    concepts_to_cover: List[str]
    resources: List[LearningResource]
    difficulty_progression: List[float]
    break_intervals: List[int]  # Minutes into session for breaks
    personalization_score: float
    expected_learning_gain: float

@dataclass
class PeerGroup:
    """Recommended peer group for collaborative learning"""
    group_id: str
    group_name: str
    members: List[str]         # Student IDs
    compatibility_score: float
    shared_topics: List[str]
    complementary_strengths: Dict[str, List[str]]  # Topic -> students strong in it
    learning_objectives: List[str]
    recommended_activities: List[str]

@dataclass
class Recommendation:
    """Generic recommendation with explanation"""
    recommendation_id: str
    student_id: str
    recommendation_type: RecommendationType
    item_id: str               # ID of recommended item
    title: str
    description: str
    confidence_score: float    # 0-1 scale
    relevance_score: float     # How relevant to current learning
    personalization_score: float  # How personalized to student
    reasoning: RecommendationReason
    explanation: str           # Human-readable explanation
    supporting_evidence: Dict[str, Any]
    expected_benefit: str
    estimated_time: int        # Minutes
    priority_level: int        # 1-5 scale
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(days=7))
    created_at: datetime = field(default_factory=datetime.now)

class CollaborativeFilteringEngine:
    """Collaborative filtering for educational recommendations"""
    
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.user_embeddings = {}
        self.item_embeddings = {}
        self.interaction_matrix = None
        self.model = None
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.is_fitted = False
    
    def prepare_interaction_data(self, interactions: pd.DataFrame) -> np.ndarray:
        """Prepare interaction data for collaborative filtering"""
        try:
            # Create user and item mappings
            unique_users = interactions['user_id'].unique()
            unique_items = interactions['item_id'].unique()
            
            self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
            self.item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
            
            # Create interaction matrix
            n_users = len(unique_users)
            n_items = len(unique_items)
            
            interaction_matrix = np.zeros((n_users, n_items))
            
            for _, row in interactions.iterrows():
                user_idx = self.user_to_idx[row['user_id']]
                item_idx = self.item_to_idx[row['item_id']]
                
                # Use success rate or engagement as rating
                rating = row.get('success_rate', 0.5)
                if 'engagement_time' in row:
                    # Normalize engagement time to 0-1 scale
                    rating = (rating + min(1.0, row['engagement_time'] / 3600)) / 2
                
                interaction_matrix[user_idx, item_idx] = rating
            
            self.interaction_matrix = interaction_matrix
            return interaction_matrix
            
        except Exception as e:
            logger.error(f"❌ Failed to prepare interaction data: {e}")
            return np.array([])
    
    def fit_matrix_factorization(self, interaction_matrix: np.ndarray, 
                               n_factors: int = 50, n_iter: int = 100):
        """Fit matrix factorization model"""
        try:
            # Use Non-negative Matrix Factorization
            self.model = NMF(n_components=n_factors, max_iter=n_iter, random_state=42)
            
            # Fit the model
            W = self.model.fit_transform(interaction_matrix)  # User factors
            H = self.model.components_  # Item factors
            
            # Store embeddings
            for user_id, idx in self.user_to_idx.items():
                self.user_embeddings[user_id] = W[idx]
            
            for item_id, idx in self.item_to_idx.items():
                self.item_embeddings[item_id] = H[:, idx]
            
            self.is_fitted = True
            logger.info(f"✅ Matrix factorization fitted with {n_factors} factors")
            
        except Exception as e:
            logger.error(f"❌ Failed to fit matrix factorization: {e}")
    
    def get_user_recommendations(self, user_id: str, n_recommendations: int = 10,
                               exclude_seen: bool = True) -> List[Tuple[str, float]]:
        """Get recommendations for a user"""
        try:
            if not self.is_fitted or user_id not in self.user_embeddings:
                return []
            
            user_embedding = self.user_embeddings[user_id]
            recommendations = []
            
            # Calculate scores for all items
            for item_id, item_embedding in self.item_embeddings.items():
                score = np.dot(user_embedding, item_embedding)
                recommendations.append((item_id, score))
            
            # Sort by score
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            # Filter out already seen items if requested
            if exclude_seen and user_id in self.user_to_idx:
                user_idx = self.user_to_idx[user_id]
                seen_items = set()
                
                for item_id, item_idx in self.item_to_idx.items():
                    if self.interaction_matrix[user_idx, item_idx] > 0:
                        seen_items.add(item_id)
                
                recommendations = [(item_id, score) for item_id, score in recommendations
                                 if item_id not in seen_items]
            
            return recommendations[:n_recommendations]
            
        except Exception as e:
            logger.error(f"❌ Failed to get user recommendations: {e}")
            return []
    
    def find_similar_users(self, user_id: str, n_similar: int = 10) -> List[Tuple[str, float]]:
        """Find users similar to the given user"""
        try:
            if not self.is_fitted or user_id not in self.user_embeddings:
                return []
            
            user_embedding = self.user_embeddings[user_id]
            similarities = []
            
            for other_user_id, other_embedding in self.user_embeddings.items():
                if other_user_id != user_id:
                    # Calculate cosine similarity
                    norm_user = np.linalg.norm(user_embedding)
                    norm_other = np.linalg.norm(other_embedding)
                    
                    if norm_user > 0 and norm_other > 0:
                        similarity = np.dot(user_embedding, other_embedding) / (norm_user * norm_other)
                        similarities.append((other_user_id, similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:n_similar]
            
        except Exception as e:
            logger.error(f"❌ Failed to find similar users: {e}")
            return []

class ContentBasedEngine:
    """Content-based filtering using learning resource features"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.content_features = {}
        self.similarity_matrix = None
        self.resource_ids = []
        self.is_fitted = False
    
    def fit(self, resources: List[LearningResource]):
        """Fit content-based model on learning resources"""
        try:
            self.resource_ids = [resource.resource_id for resource in resources]
            
            # Combine text features
            content_texts = []
            for resource in resources:
                text_features = [
                    resource.title,
                    resource.topic,
                    ' '.join(resource.learning_objectives),
                    ' '.join(resource.tags)
                ]
                content_texts.append(' '.join(text_features))
            
            # Create TF-IDF features
            tfidf_features = self.vectorizer.fit_transform(content_texts)
            
            # Add numerical features
            numerical_features = []
            for resource in resources:
                features = [
                    resource.difficulty_level,
                    resource.duration_minutes / 120.0,  # Normalize to ~0-1 range
                    resource.quality_score,
                    resource.engagement_score,
                    resource.effectiveness_score
                ]
                numerical_features.append(features)
            
            numerical_features = np.array(numerical_features)
            
            # Normalize numerical features
            scaler = StandardScaler()
            numerical_features = scaler.fit_transform(numerical_features)
            
            # Combine TF-IDF and numerical features
            combined_features = np.hstack([tfidf_features.toarray(), numerical_features])
            
            # Calculate similarity matrix
            self.similarity_matrix = cosine_similarity(combined_features)
            
            # Store features for each resource
            for i, resource in enumerate(resources):
                self.content_features[resource.resource_id] = combined_features[i]
            
            self.is_fitted = True
            logger.info(f"✅ Content-based model fitted on {len(resources)} resources")
            
        except Exception as e:
            logger.error(f"❌ Failed to fit content-based model: {e}")
    
    def get_similar_resources(self, resource_id: str, n_similar: int = 10) -> List[Tuple[str, float]]:
        """Get resources similar to the given resource"""
        try:
            if not self.is_fitted or resource_id not in self.resource_ids:
                return []
            
            resource_idx = self.resource_ids.index(resource_id)
            similarities = self.similarity_matrix[resource_idx]
            
            # Get top similar resources (excluding the resource itself)
            similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]
            
            similar_resources = []
            for idx in similar_indices:
                similar_resource_id = self.resource_ids[idx]
                similarity_score = similarities[idx]
                similar_resources.append((similar_resource_id, similarity_score))
            
            return similar_resources
            
        except Exception as e:
            logger.error(f"❌ Failed to get similar resources: {e}")
            return []
    
    def recommend_for_profile(self, student_profile: Dict[str, Any], 
                            available_resources: List[str],
                            n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """Recommend resources based on student profile"""
        try:
            if not self.is_fitted:
                return []
            
            recommendations = []
            
            # Extract profile preferences
            preferred_difficulty = student_profile.get('preferred_difficulty', 0.5)
            learning_style = student_profile.get('learning_style', 'mixed')
            strong_topics = student_profile.get('strong_topics', [])
            weak_topics = student_profile.get('weak_topics', [])
            
            for resource_id in available_resources:
                if resource_id not in self.content_features:
                    continue
                
                # Calculate compatibility score
                score = 0.0
                
                # Difficulty match (prefer slightly above current level)
                resource_features = self.content_features[resource_id]
                # Assuming difficulty is one of the numerical features
                # This would need to be adjusted based on actual feature ordering
                
                # For now, use a simple scoring based on available information
                score += 0.5  # Base score
                
                # Boost score for weak topics (remedial content)
                # and reduce for already strong topics
                # This would require topic mapping in the feature space
                
                recommendations.append((resource_id, score))
            
            # Sort by score
            recommendations.sort(key=lambda x: x[1], reverse=True)
            return recommendations[:n_recommendations]
            
        except Exception as e:
            logger.error(f"❌ Failed to recommend for profile: {e}")
            return []

class HybridRecommendationEngine:
    """Hybrid recommendation engine combining multiple approaches"""
    
    def __init__(self, db_manager=None, redis_client=None):
        self.db_manager = db_manager
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, db=0)
        
        # Component engines
        self.collaborative_engine = CollaborativeFilteringEngine()
        self.content_engine = ContentBasedEngine()
        
        # Knowledge graph for educational relationships
        self.knowledge_graph = None
        
        # Student models and preferences
        self.student_profiles = {}
        self.learning_resources = {}
        self.interaction_history = defaultdict(list)
        
        # Recommendation weights
        self.weights = {
            'collaborative': 0.3,
            'content_based': 0.25,
            'knowledge_graph': 0.2,
            'performance_based': 0.15,
            'temporal': 0.1
        }
        
        # Caching
        self.cache_ttl = 3600  # 1 hour
        
        # A/B testing
        self.recommendation_variants = {}
        self.variant_performance = defaultdict(list)
    
    async def initialize(self):
        """Initialize the hybrid recommendation system"""
        try:
            logger.info("🚀 Initializing Personalized Recommendation System")
            
            # Load learning resources
            await self._load_learning_resources()
            
            # Load student profiles
            await self._load_student_profiles()
            
            # Load interaction history
            await self._load_interaction_history()
            
            # Load knowledge graph
            await self._load_knowledge_graph()
            
            # Train collaborative filtering
            await self._train_collaborative_filtering()
            
            # Train content-based filtering
            await self._train_content_based_filtering()
            
            logger.info("✅ Personalized Recommendation System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Personalized Recommendation System: {e}")
            return False
    
    async def _load_learning_resources(self):
        """Load learning resources from database"""
        try:
            # Sample learning resources for testing
            self.learning_resources = {
                "physics_basics_video": LearningResource(
                    resource_id="physics_basics_video",
                    title="Introduction to Physics Concepts",
                    content_type="video",
                    topic="physics_basics",
                    difficulty_level=0.2,
                    duration_minutes=30,
                    learning_objectives=["understand basic physics", "prepare for kinematics"],
                    prerequisites=[],
                    tags=["introductory", "conceptual", "visual"],
                    quality_score=0.85,
                    engagement_score=0.9,
                    effectiveness_score=0.8
                ),
                "kinematics_problems": LearningResource(
                    resource_id="kinematics_problems",
                    title="1D Kinematics Problem Set",
                    content_type="problem",
                    topic="kinematics",
                    difficulty_level=0.5,
                    duration_minutes=45,
                    learning_objectives=["solve motion problems", "apply kinematic equations"],
                    prerequisites=["physics_basics"],
                    tags=["problem_solving", "equations", "practice"],
                    quality_score=0.8,
                    engagement_score=0.7,
                    effectiveness_score=0.85
                ),
                "forces_simulation": LearningResource(
                    resource_id="forces_simulation",
                    title="Interactive Force Simulation",
                    content_type="interactive",
                    topic="forces",
                    difficulty_level=0.6,
                    duration_minutes=25,
                    learning_objectives=["visualize forces", "understand vector addition"],
                    prerequisites=["kinematics"],
                    tags=["interactive", "visualization", "forces"],
                    quality_score=0.9,
                    engagement_score=0.95,
                    effectiveness_score=0.8
                ),
                "energy_derivation": LearningResource(
                    resource_id="energy_derivation",
                    title="Energy Conservation Derivation",
                    content_type="text",
                    topic="energy",
                    difficulty_level=0.7,
                    duration_minutes=20,
                    learning_objectives=["derive energy equations", "understand conservation"],
                    prerequisites=["forces"],
                    tags=["mathematical", "derivation", "theory"],
                    quality_score=0.75,
                    engagement_score=0.6,
                    effectiveness_score=0.9
                )
            }
            
            if self.db_manager:
                # In a real implementation, load from database
                async with self.db_manager.postgres.get_connection() as conn:
                    resources = await conn.fetch("""
                        SELECT * FROM learning_resources 
                        WHERE is_active = TRUE
                    """)
                    
                    for resource in resources:
                        # Convert database record to LearningResource object
                        pass  # Implementation would depend on database schema
            
            logger.info(f"📚 Loaded {len(self.learning_resources)} learning resources")
            
        except Exception as e:
            logger.error(f"❌ Failed to load learning resources: {e}")
    
    async def _load_student_profiles(self):
        """Load student profiles and preferences"""
        try:
            if not self.db_manager:
                # Create sample profiles for testing
                self.student_profiles = {
                    "test_student": {
                        'preferred_difficulty': 0.6,
                        'learning_style': 'visual',
                        'strong_topics': ['kinematics'],
                        'weak_topics': ['forces'],
                        'preferred_duration': 30,
                        'study_times': [14, 15, 16],  # 2-4 PM
                        'engagement_preferences': ['interactive', 'visual']
                    }
                }
                return
            
            async with self.db_manager.postgres.get_connection() as conn:
                students = await conn.fetch("""
                    SELECT u.id, u.username, up.topic, up.proficiency_score
                    FROM users u
                    LEFT JOIN user_progress up ON u.id = up.user_id
                    WHERE u.is_active = TRUE
                """)
                
                # Group by student and build profiles
                student_data = defaultdict(dict)
                for row in students:
                    student_id = str(row['id'])
                    if row['topic'] and row['proficiency_score'] is not None:
                        student_data[student_id][row['topic']] = row['proficiency_score'] / 100.0
                
                # Build student profiles
                for student_id, topics in student_data.items():
                    avg_performance = np.mean(list(topics.values())) if topics else 0.5
                    strong_topics = [topic for topic, score in topics.items() if score > 0.7]
                    weak_topics = [topic for topic, score in topics.items() if score < 0.5]
                    
                    self.student_profiles[student_id] = {
                        'preferred_difficulty': min(0.8, avg_performance + 0.1),
                        'learning_style': 'mixed',  # Would be detected from interactions
                        'strong_topics': strong_topics,
                        'weak_topics': weak_topics,
                        'preferred_duration': 30,
                        'study_times': [14, 15, 16],
                        'engagement_preferences': ['interactive']
                    }
            
            logger.info(f"👥 Loaded profiles for {len(self.student_profiles)} students")
            
        except Exception as e:
            logger.error(f"❌ Failed to load student profiles: {e}")
    
    async def _load_interaction_history(self):
        """Load student interaction history"""
        try:
            if not self.db_manager:
                # Create sample interaction data
                sample_interactions = [
                    {'user_id': 'test_student', 'item_id': 'physics_basics_video', 'success_rate': 0.8, 'engagement_time': 1800},
                    {'user_id': 'test_student', 'item_id': 'kinematics_problems', 'success_rate': 0.6, 'engagement_time': 2400},
                ]
                
                for interaction in sample_interactions:
                    self.interaction_history[interaction['user_id']].append(interaction)
                return
            
            async with self.db_manager.postgres.get_connection() as conn:
                interactions = await conn.fetch("""
                    SELECT user_id, agent_type as item_id,
                           AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate,
                           AVG(execution_time_ms) / 1000.0 as engagement_time
                    FROM interactions 
                    WHERE created_at >= $1
                    GROUP BY user_id, agent_type
                """, datetime.now() - timedelta(days=30))
                
                for interaction in interactions:
                    student_id = str(interaction['user_id'])
                    self.interaction_history[student_id].append({
                        'item_id': interaction['item_id'],
                        'success_rate': interaction['success_rate'],
                        'engagement_time': interaction['engagement_time']
                    })
            
            logger.info(f"📊 Loaded interaction history for {len(self.interaction_history)} students")
            
        except Exception as e:
            logger.error(f"❌ Failed to load interaction history: {e}")
    
    async def _load_knowledge_graph(self):
        """Load knowledge graph for educational relationships"""
        try:
            self.knowledge_graph = nx.DiGraph()
            
            # Sample knowledge graph
            concepts = ["physics_basics", "kinematics", "forces", "energy", "momentum"]
            for concept in concepts:
                self.knowledge_graph.add_node(concept)
            
            # Add prerequisite relationships
            edges = [
                ("physics_basics", "kinematics"),
                ("kinematics", "forces"),
                ("forces", "energy"),
                ("forces", "momentum")
            ]
            self.knowledge_graph.add_edges_from(edges)
            
            if self.db_manager:
                # Load from Neo4j knowledge graph
                concepts_query = """
                MATCH (c:Concept)
                RETURN c.name as name
                """
                
                prereq_query = """
                MATCH (prereq:Concept)-[:PREREQUISITE]->(concept:Concept)
                RETURN prereq.name as prerequisite, concept.name as concept
                """
                
                concepts = await self.db_manager.neo4j.run_query(concepts_query)
                prerequisites = await self.db_manager.neo4j.run_query(prereq_query)
                
                # Build graph
                for concept in concepts:
                    self.knowledge_graph.add_node(concept['name'])
                
                for prereq in prerequisites:
                    self.knowledge_graph.add_edge(prereq['prerequisite'], prereq['concept'])
            
            logger.info(f"🕸️ Loaded knowledge graph with {len(self.knowledge_graph.nodes())} concepts")
            
        except Exception as e:
            logger.error(f"❌ Failed to load knowledge graph: {e}")
    
    async def _train_collaborative_filtering(self):
        """Train collaborative filtering model"""
        try:
            # Prepare interaction data
            interaction_data = []
            for student_id, interactions in self.interaction_history.items():
                for interaction in interactions:
                    interaction_data.append({
                        'user_id': student_id,
                        'item_id': interaction['item_id'],
                        'success_rate': interaction.get('success_rate', 0.5),
                        'engagement_time': interaction.get('engagement_time', 0)
                    })
            
            if not interaction_data:
                logger.warning("⚠️ No interaction data available for collaborative filtering")
                return
            
            interaction_df = pd.DataFrame(interaction_data)
            interaction_matrix = self.collaborative_engine.prepare_interaction_data(interaction_df)
            
            if interaction_matrix.size > 0:
                self.collaborative_engine.fit_matrix_factorization(interaction_matrix)
                logger.info("✅ Collaborative filtering model trained")
            
        except Exception as e:
            logger.error(f"❌ Failed to train collaborative filtering: {e}")
    
    async def _train_content_based_filtering(self):
        """Train content-based filtering model"""
        try:
            resources_list = list(self.learning_resources.values())
            if resources_list:
                self.content_engine.fit(resources_list)
                logger.info("✅ Content-based filtering model trained")
            
        except Exception as e:
            logger.error(f"❌ Failed to train content-based filtering: {e}")
    
    async def generate_recommendations(self, student_id: str, 
                                     recommendation_type: RecommendationType = RecommendationType.CONTENT,
                                     n_recommendations: int = 10,
                                     context: Dict[str, Any] = None) -> List[Recommendation]:
        """Generate personalized recommendations for a student"""
        try:
            logger.info(f"🎯 Generating {recommendation_type.value} recommendations for {student_id}")
            
            # Check cache first
            cache_key = f"recommendations:{student_id}:{recommendation_type.value}:{n_recommendations}"
            cached_recommendations = await self._get_cached_recommendations(cache_key)
            if cached_recommendations:
                return cached_recommendations
            
            # Get student profile
            student_profile = self.student_profiles.get(student_id, {})
            
            # Generate recommendations based on type
            if recommendation_type == RecommendationType.CONTENT:
                recommendations = await self._generate_content_recommendations(
                    student_id, student_profile, n_recommendations, context
                )
            elif recommendation_type == RecommendationType.LEARNING_PATH:
                recommendations = await self._generate_learning_path_recommendations(
                    student_id, student_profile, n_recommendations, context
                )
            elif recommendation_type == RecommendationType.STUDY_SCHEDULE:
                recommendations = await self._generate_study_schedule_recommendations(
                    student_id, student_profile, n_recommendations, context
                )
            elif recommendation_type == RecommendationType.PEER_GROUP:
                recommendations = await self._generate_peer_group_recommendations(
                    student_id, student_profile, n_recommendations, context
                )
            else:
                recommendations = await self._generate_content_recommendations(
                    student_id, student_profile, n_recommendations, context
                )
            
            # Cache recommendations
            await self._cache_recommendations(cache_key, recommendations)
            
            logger.info(f"✅ Generated {len(recommendations)} recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Failed to generate recommendations: {e}")
            return []
    
    async def _generate_content_recommendations(self, student_id: str, 
                                              student_profile: Dict[str, Any],
                                              n_recommendations: int,
                                              context: Dict[str, Any] = None) -> List[Recommendation]:
        """Generate content recommendations using hybrid approach"""
        try:
            recommendations = []
            recommendation_scores = defaultdict(list)
            
            # 1. Collaborative Filtering Recommendations
            if self.collaborative_engine.is_fitted:
                cf_recommendations = self.collaborative_engine.get_user_recommendations(
                    student_id, n_recommendations * 2
                )
                
                for item_id, score in cf_recommendations:
                    if item_id in self.learning_resources:
                        recommendation_scores[item_id].append({
                            'method': 'collaborative',
                            'score': score,
                            'weight': self.weights['collaborative']
                        })
            
            # 2. Content-Based Recommendations
            if self.content_engine.is_fitted:
                # Find resources similar to ones the student liked
                liked_resources = []
                for interaction in self.interaction_history.get(student_id, []):
                    if interaction.get('success_rate', 0) > 0.7:
                        liked_resources.append(interaction['item_id'])
                
                for liked_resource in liked_resources[:3]:  # Top 3 liked
                    similar_resources = self.content_engine.get_similar_resources(
                        liked_resource, n_recommendations
                    )
                    
                    for item_id, score in similar_resources:
                        if item_id in self.learning_resources:
                            recommendation_scores[item_id].append({
                                'method': 'content_based',
                                'score': score,
                                'weight': self.weights['content_based']
                            })
                
                # Also get recommendations based on profile
                available_resources = list(self.learning_resources.keys())
                profile_recommendations = self.content_engine.recommend_for_profile(
                    student_profile, available_resources, n_recommendations
                )
                
                for item_id, score in profile_recommendations:
                    recommendation_scores[item_id].append({
                        'method': 'profile_based',
                        'score': score,
                        'weight': self.weights['content_based']
                    })
            
            # 3. Knowledge Graph Recommendations
            knowledge_recommendations = await self._get_knowledge_graph_recommendations(
                student_id, student_profile, n_recommendations
            )
            
            for item_id, score in knowledge_recommendations:
                recommendation_scores[item_id].append({
                    'method': 'knowledge_graph',
                    'score': score,
                    'weight': self.weights['knowledge_graph']
                })
            
            # 4. Performance-Based Recommendations
            performance_recommendations = await self._get_performance_based_recommendations(
                student_id, student_profile, n_recommendations
            )
            
            for item_id, score in performance_recommendations:
                recommendation_scores[item_id].append({
                    'method': 'performance',
                    'score': score,
                    'weight': self.weights['performance_based']
                })
            
            # Combine scores using weighted average
            final_scores = {}
            for item_id, scores in recommendation_scores.items():
                weighted_score = sum(s['score'] * s['weight'] for s in scores) / sum(s['weight'] for s in scores)
                final_scores[item_id] = {
                    'score': weighted_score,
                    'methods': [s['method'] for s in scores]
                }
            
            # Sort by final score
            sorted_items = sorted(final_scores.items(), key=lambda x: x[1]['score'], reverse=True)
            
            # Create Recommendation objects
            for i, (item_id, score_info) in enumerate(sorted_items[:n_recommendations]):
                resource = self.learning_resources[item_id]
                
                # Determine primary reasoning
                primary_method = Counter(score_info['methods']).most_common(1)[0][0]
                reasoning_map = {
                    'collaborative': RecommendationReason.COLLABORATIVE_FILTERING,
                    'content_based': RecommendationReason.CONTENT_BASED,
                    'knowledge_graph': RecommendationReason.KNOWLEDGE_GAP,
                    'performance': RecommendationReason.PERFORMANCE_BASED
                }
                reasoning = reasoning_map.get(primary_method, RecommendationReason.CONTENT_BASED)
                
                # Generate explanation
                explanation = self._generate_explanation(resource, reasoning, student_profile)
                
                recommendation = Recommendation(
                    recommendation_id=f"rec_{student_id}_{item_id}_{datetime.now().timestamp()}",
                    student_id=student_id,
                    recommendation_type=RecommendationType.CONTENT,
                    item_id=item_id,
                    title=resource.title,
                    description=f"A {resource.content_type} on {resource.topic}",
                    confidence_score=min(1.0, score_info['score']),
                    relevance_score=self._calculate_relevance_score(resource, student_profile),
                    personalization_score=len(score_info['methods']) / 4.0,  # More methods = more personalized
                    reasoning=reasoning,
                    explanation=explanation,
                    supporting_evidence={
                        'methods_used': score_info['methods'],
                        'combined_score': score_info['score'],
                        'difficulty_match': abs(resource.difficulty_level - student_profile.get('preferred_difficulty', 0.5))
                    },
                    expected_benefit=f"Improve understanding of {resource.topic}",
                    estimated_time=resource.duration_minutes,
                    priority_level=min(5, max(1, int(score_info['score'] * 5) + 1))
                )
                
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Failed to generate content recommendations: {e}")
            return []
    
    async def _get_knowledge_graph_recommendations(self, student_id: str,
                                                 student_profile: Dict[str, Any],
                                                 n_recommendations: int) -> List[Tuple[str, float]]:
        """Get recommendations based on knowledge graph relationships"""
        try:
            recommendations = []
            
            if not self.knowledge_graph:
                return recommendations
            
            # Identify next concepts to learn based on prerequisites
            mastered_topics = student_profile.get('strong_topics', [])
            weak_topics = student_profile.get('weak_topics', [])
            
            # Find concepts that the student is ready to learn
            ready_concepts = []
            for concept in self.knowledge_graph.nodes():
                if concept not in mastered_topics:
                    # Check if prerequisites are satisfied
                    prerequisites = list(self.knowledge_graph.predecessors(concept))
                    if all(prereq in mastered_topics for prereq in prerequisites):
                        ready_concepts.append(concept)
            
            # Score resources based on readiness
            for resource_id, resource in self.learning_resources.items():
                score = 0.0
                
                # High score for ready concepts
                if resource.topic in ready_concepts:
                    score += 0.8
                
                # Medium score for weak topics (remedial)
                elif resource.topic in weak_topics:
                    score += 0.6
                
                # Lower score for already mastered topics
                elif resource.topic in mastered_topics:
                    score += 0.2
                
                # Bonus for appropriate difficulty
                preferred_difficulty = student_profile.get('preferred_difficulty', 0.5)
                difficulty_match = 1.0 - abs(resource.difficulty_level - preferred_difficulty)
                score += difficulty_match * 0.2
                
                if score > 0:
                    recommendations.append((resource_id, score))
            
            recommendations.sort(key=lambda x: x[1], reverse=True)
            return recommendations[:n_recommendations]
            
        except Exception as e:
            logger.error(f"❌ Failed to get knowledge graph recommendations: {e}")
            return []
    
    async def _get_performance_based_recommendations(self, student_id: str,
                                                   student_profile: Dict[str, Any],
                                                   n_recommendations: int) -> List[Tuple[str, float]]:
        """Get recommendations based on performance patterns"""
        try:
            recommendations = []
            
            # Analyze student's performance patterns
            interactions = self.interaction_history.get(student_id, [])
            if not interactions:
                return recommendations
            
            # Calculate average performance by topic
            topic_performance = defaultdict(list)
            for interaction in interactions:
                # Map item_id to topic (simplified)
                resource = self.learning_resources.get(interaction['item_id'])
                if resource:
                    topic_performance[resource.topic].append(interaction.get('success_rate', 0.5))
            
            topic_averages = {topic: np.mean(scores) for topic, scores in topic_performance.items()}
            
            # Recommend resources for topics with declining performance
            for resource_id, resource in self.learning_resources.items():
                score = 0.0
                
                topic_avg = topic_averages.get(resource.topic, 0.5)
                
                # Higher score for topics with lower performance (need improvement)
                if topic_avg < 0.6:
                    score += (0.6 - topic_avg) * 2.0
                
                # Consider resource effectiveness
                score += resource.effectiveness_score * 0.3
                
                # Prefer easier resources for struggling topics
                if topic_avg < 0.5 and resource.difficulty_level < 0.5:
                    score += 0.4
                
                if score > 0:
                    recommendations.append((resource_id, score))
            
            recommendations.sort(key=lambda x: x[1], reverse=True)
            return recommendations[:n_recommendations]
            
        except Exception as e:
            logger.error(f"❌ Failed to get performance-based recommendations: {e}")
            return []
    
    def _calculate_relevance_score(self, resource: LearningResource, 
                                 student_profile: Dict[str, Any]) -> float:
        """Calculate how relevant a resource is to the student"""
        try:
            score = 0.5  # Base relevance
            
            # Topic relevance
            weak_topics = student_profile.get('weak_topics', [])
            strong_topics = student_profile.get('strong_topics', [])
            
            if resource.topic in weak_topics:
                score += 0.3  # Highly relevant for improvement
            elif resource.topic in strong_topics:
                score += 0.1  # Somewhat relevant for advancement
            
            # Difficulty appropriateness
            preferred_difficulty = student_profile.get('preferred_difficulty', 0.5)
            difficulty_match = 1.0 - abs(resource.difficulty_level - preferred_difficulty)
            score += difficulty_match * 0.2
            
            # Duration preference
            preferred_duration = student_profile.get('preferred_duration', 30)
            if abs(resource.duration_minutes - preferred_duration) <= 10:
                score += 0.1
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate relevance score: {e}")
            return 0.5
    
    def _generate_explanation(self, resource: LearningResource, 
                            reasoning: RecommendationReason,
                            student_profile: Dict[str, Any]) -> str:
        """Generate human-readable explanation for recommendation"""
        try:
            explanations = {
                RecommendationReason.COLLABORATIVE_FILTERING: 
                    f"Students with similar learning patterns found this {resource.content_type} helpful for {resource.topic}.",
                RecommendationReason.CONTENT_BASED:
                    f"This {resource.content_type} matches your interests in {resource.topic} and learning style preferences.",
                RecommendationReason.KNOWLEDGE_GAP:
                    f"This resource addresses {resource.topic}, which is recommended based on your current knowledge state.",
                RecommendationReason.PERFORMANCE_BASED:
                    f"This resource can help improve your performance in {resource.topic} based on your recent activity.",
                RecommendationReason.LEARNING_STYLE:
                    f"This {resource.content_type} format aligns with your preferred learning style."
            }
            
            base_explanation = explanations.get(reasoning, "This resource is recommended based on your learning profile.")
            
            # Add difficulty context
            if resource.difficulty_level < 0.4:
                base_explanation += " It's designed for beginners and will help build foundational understanding."
            elif resource.difficulty_level > 0.7:
                base_explanation += " It's advanced content that will challenge and expand your knowledge."
            
            return base_explanation
            
        except Exception as e:
            logger.error(f"❌ Failed to generate explanation: {e}")
            return "This resource is recommended for your learning journey."
    
    async def _generate_learning_path_recommendations(self, student_id: str,
                                                    student_profile: Dict[str, Any],
                                                    n_recommendations: int,
                                                    context: Dict[str, Any] = None) -> List[Recommendation]:
        """Generate learning path recommendations"""
        try:
            recommendations = []
            
            # Simplified learning path generation
            # In practice, this would use the adaptive learning system
            
            weak_topics = student_profile.get('weak_topics', [])
            strong_topics = student_profile.get('strong_topics', [])
            
            for i, topic in enumerate(weak_topics[:n_recommendations]):
                # Find resources for this topic
                topic_resources = [r for r in self.learning_resources.values() if r.topic == topic]
                
                if topic_resources:
                    # Sort by difficulty
                    topic_resources.sort(key=lambda x: x.difficulty_level)
                    
                    resource = topic_resources[0]  # Start with easiest
                    
                    recommendation = Recommendation(
                        recommendation_id=f"path_{student_id}_{topic}_{datetime.now().timestamp()}",
                        student_id=student_id,
                        recommendation_type=RecommendationType.LEARNING_PATH,
                        item_id=f"path_{topic}",
                        title=f"Learning Path: {topic.title()}",
                        description=f"Structured learning sequence for mastering {topic}",
                        confidence_score=0.8,
                        relevance_score=0.9,
                        personalization_score=0.7,
                        reasoning=RecommendationReason.KNOWLEDGE_GAP,
                        explanation=f"This learning path will help you master {topic}, starting with foundational concepts.",
                        supporting_evidence={'weak_topic': True, 'structured_approach': True},
                        expected_benefit=f"Achieve mastery in {topic}",
                        estimated_time=sum(r.duration_minutes for r in topic_resources),
                        priority_level=5 - i  # Higher priority for first topics
                    )
                    
                    recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Failed to generate learning path recommendations: {e}")
            return []
    
    async def _generate_study_schedule_recommendations(self, student_id: str,
                                                     student_profile: Dict[str, Any],
                                                     n_recommendations: int,
                                                     context: Dict[str, Any] = None) -> List[Recommendation]:
        """Generate study schedule recommendations"""
        try:
            recommendations = []
            
            # Get preferred study times
            preferred_times = student_profile.get('study_times', [14, 15, 16])
            preferred_duration = student_profile.get('preferred_duration', 30)
            
            # Create study session recommendations
            for i in range(min(n_recommendations, 3)):
                study_time = preferred_times[i % len(preferred_times)]
                
                recommendation = Recommendation(
                    recommendation_id=f"schedule_{student_id}_{study_time}_{datetime.now().timestamp()}",
                    student_id=student_id,
                    recommendation_type=RecommendationType.STUDY_SCHEDULE,
                    item_id=f"session_{study_time}",
                    title=f"Study Session at {study_time}:00",
                    description=f"Optimized {preferred_duration}-minute study session",
                    confidence_score=0.7,
                    relevance_score=0.8,
                    personalization_score=0.9,
                    reasoning=RecommendationReason.TEMPORAL_PATTERN,
                    explanation=f"Based on your study patterns, {study_time}:00 is an optimal time for learning.",
                    supporting_evidence={'preferred_time': True, 'duration_match': True},
                    expected_benefit="Improved focus and retention",
                    estimated_time=preferred_duration,
                    priority_level=3
                )
                
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Failed to generate study schedule recommendations: {e}")
            return []
    
    async def _generate_peer_group_recommendations(self, student_id: str,
                                                 student_profile: Dict[str, Any],
                                                 n_recommendations: int,
                                                 context: Dict[str, Any] = None) -> List[Recommendation]:
        """Generate peer group recommendations"""
        try:
            recommendations = []
            
            # Find similar students using collaborative filtering
            similar_students = self.collaborative_engine.find_similar_users(student_id, 10)
            
            if len(similar_students) >= 2:
                # Create peer group recommendation
                group_members = [s[0] for s in similar_students[:3]]  # Top 3 similar students
                
                recommendation = Recommendation(
                    recommendation_id=f"peer_group_{student_id}_{datetime.now().timestamp()}",
                    student_id=student_id,
                    recommendation_type=RecommendationType.PEER_GROUP,
                    item_id=f"group_{hashlib.md5('_'.join(group_members).encode()).hexdigest()[:8]}",
                    title="Study Group Recommendation",
                    description="Students with similar learning patterns and goals",
                    confidence_score=0.75,
                    relevance_score=0.8,
                    personalization_score=0.9,
                    reasoning=RecommendationReason.PEER_SUCCESS,
                    explanation="These students have similar learning patterns and could benefit from collaborative study.",
                    supporting_evidence={'similarity_scores': [s[1] for s in similar_students[:3]]},
                    expected_benefit="Enhanced learning through peer collaboration",
                    estimated_time=60,  # Typical group study session
                    priority_level=3
                )
                
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Failed to generate peer group recommendations: {e}")
            return []
    
    async def _get_cached_recommendations(self, cache_key: str) -> Optional[List[Recommendation]]:
        """Get cached recommendations if available"""
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                # In practice, would deserialize properly
                return []  # Placeholder
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get cached recommendations: {e}")
            return None
    
    async def _cache_recommendations(self, cache_key: str, recommendations: List[Recommendation]):
        """Cache recommendations for future use"""
        try:
            # In practice, would serialize recommendations properly
            self.redis_client.setex(cache_key, self.cache_ttl, "cached_data")
            
        except Exception as e:
            logger.error(f"❌ Failed to cache recommendations: {e}")
    
    async def record_interaction(self, student_id: str, recommendation_id: str, 
                               interaction_type: str, outcome: Dict[str, Any]):
        """Record interaction with recommendation for feedback learning"""
        try:
            # Store interaction for model improvement
            interaction_data = {
                'student_id': student_id,
                'recommendation_id': recommendation_id,
                'interaction_type': interaction_type,  # 'viewed', 'clicked', 'completed', 'dismissed'
                'outcome': outcome,
                'timestamp': datetime.now()
            }
            
            # Update recommendation effectiveness
            if interaction_type == 'completed' and outcome.get('success', False):
                # Positive feedback - recommendation was effective
                pass
            elif interaction_type == 'dismissed':
                # Negative feedback - recommendation was not relevant
                pass
            
            logger.info(f"📊 Recorded recommendation interaction: {interaction_type}")
            
        except Exception as e:
            logger.error(f"❌ Failed to record interaction: {e}")
    
    async def update_student_preferences(self, student_id: str, preferences: Dict[str, Any]):
        """Update student preferences for better recommendations"""
        try:
            if student_id not in self.student_profiles:
                self.student_profiles[student_id] = {}
            
            self.student_profiles[student_id].update(preferences)
            
            # Invalidate cache for this student
            cache_pattern = f"recommendations:{student_id}:*"
            # In practice, would use Redis pattern deletion
            
            logger.info(f"👤 Updated preferences for student {student_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to update student preferences: {e}")

# Testing function
async def test_personalized_recommendation_system():
    """Test the personalized recommendation system"""
    try:
        logger.info("🧪 Testing Personalized Recommendation System")
        
        system = HybridRecommendationEngine()
        await system.initialize()
        
        # Test content recommendations
        content_recs = await system.generate_recommendations(
            "test_student", RecommendationType.CONTENT, 5
        )
        logger.info(f"✅ Generated {len(content_recs)} content recommendations")
        
        # Test learning path recommendations
        path_recs = await system.generate_recommendations(
            "test_student", RecommendationType.LEARNING_PATH, 3
        )
        logger.info(f"✅ Generated {len(path_recs)} learning path recommendations")
        
        # Test interaction recording
        if content_recs:
            await system.record_interaction(
                "test_student", content_recs[0].recommendation_id, 
                "completed", {"success": True, "rating": 4.5}
            )
            logger.info("✅ Recorded recommendation interaction")
        
        logger.info("✅ Personalized Recommendation System test completed")
        
    except Exception as e:
        logger.error(f"❌ Personalized Recommendation System test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_personalized_recommendation_system())