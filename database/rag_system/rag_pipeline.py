#!/usr/bin/env python3
"""
RAG Query Processing Pipeline for Physics Assistant
Orchestrates the complete RAG system including embeddings, semantic search, 
graph retrieval, and context-aware ranking
"""
import os
import json
import logging
import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import asynccontextmanager

# Third-party imports
import redis
from neo4j import GraphDatabase
import asyncpg

# Local imports
from .vector_embeddings import PhysicsEmbeddingManager, get_physics_embedding_manager
from .semantic_search import SemanticSearchEngine, SearchQuery, SearchResult, SearchType
from .graph_retrieval import GraphEnhancedRAGRetriever, GraphTraversalStrategy
from .context_aware_ranking import (
    ContextAwareRanker, StudentProfileManager, ContentAnalyzer,
    LearningContext, StudentProfile, setup_student_profile_tables
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGMode(Enum):
    """Different RAG operation modes"""
    QUICK = "quick"  # Fast semantic search only
    COMPREHENSIVE = "comprehensive"  # Full pipeline with graph enhancement
    EDUCATIONAL = "educational"  # Optimized for learning with context awareness
    RESEARCH = "research"  # Deep exploration with multiple strategies

@dataclass
class RAGQuery:
    """Comprehensive RAG query with all parameters"""
    # Basic query
    text: str
    user_id: str
    session_id: str = ""
    
    # Search parameters
    limit: int = 10
    min_similarity: float = 0.3
    content_types: List[str] = None
    
    # Mode and strategy
    mode: RAGMode = RAGMode.COMPREHENSIVE
    search_type: SearchType = SearchType.HYBRID
    traversal_strategy: GraphTraversalStrategy = GraphTraversalStrategy.BREADTH_FIRST
    
    # Educational parameters
    student_level: str = "intermediate"
    current_topic: str = ""
    learning_objective: str = ""
    include_prerequisites: bool = True
    include_related: bool = True
    
    # Personalization
    use_personalization: bool = True
    boost_struggling_concepts: bool = True
    adapt_difficulty: bool = True
    
    # Performance parameters
    max_response_time: float = 5.0  # seconds
    use_cache: bool = True
    cache_ttl: int = 1800  # 30 minutes
    
    # Output format
    include_explanations: bool = True
    include_learning_paths: bool = False
    include_related_concepts: bool = False
    format_for_llm: bool = True  # Format for LLM consumption

@dataclass
class RAGResponse:
    """Comprehensive RAG response"""
    # Query info
    query: str
    user_id: str
    session_id: str
    
    # Results
    results: List[SearchResult]
    total_results: int
    
    # Context and metadata
    processing_time: float
    cache_hit: bool = False
    strategies_used: List[str] = field(default_factory=list)
    
    # Educational insights
    learning_insights: Dict[str, Any] = field(default_factory=dict)
    personalization_applied: Dict[str, Any] = field(default_factory=dict)
    suggested_next_steps: List[str] = field(default_factory=list)
    
    # Quality metrics
    result_quality_score: float = 0.0
    coverage_score: float = 0.0  # How well the query was covered
    confidence_score: float = 0.0
    
    # Additional content
    learning_path: List[Dict] = field(default_factory=list)
    related_concepts: List[Dict] = field(default_factory=list)
    prerequisite_gaps: List[str] = field(default_factory=list)
    
    # LLM-formatted content
    formatted_context: str = ""
    reasoning_explanation: str = ""

class RAGPipeline:
    """Main RAG pipeline orchestrator"""
    
    def __init__(self, 
                 neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 postgres_config: dict,
                 redis_host: str = 'localhost', redis_port: int = 6379, redis_password: str = None,
                 embedding_models: List[str] = None):
        
        # Database configurations
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.postgres_config = postgres_config
        self.redis_config = {
            'host': redis_host,
            'port': redis_port,
            'password': redis_password
        }
        
        # Component configurations
        self.embedding_models = embedding_models or ['sentence_transformer']
        
        # Components (initialized in setup)
        self.embedding_manager = None
        self.search_engine = None
        self.graph_retriever = None
        self.profile_manager = None
        self.content_analyzer = None
        self.context_ranker = None
        
        # Database connections
        self.postgres_pool = None
        self.redis_client = None
        
        # Performance monitoring
        self.query_stats = {
            'total_queries': 0,
            'avg_response_time': 0.0,
            'cache_hit_rate': 0.0,
            'success_rate': 0.0
        }
        
        # Cache for expensive computations
        self.response_cache = {}
        
    async def initialize(self, rebuild_indices: bool = False):
        """Initialize the complete RAG pipeline"""
        logger.info("🚀 Initializing Physics Assistant RAG Pipeline")
        
        start_time = datetime.now()
        
        try:
            # Initialize database connections
            await self._initialize_databases()
            
            # Initialize embedding manager
            self.embedding_manager = PhysicsEmbeddingManager(
                self.neo4j_uri, self.neo4j_user, self.neo4j_password,
                self.redis_config['host'], self.redis_config['port'], self.redis_config['password'],
                self.postgres_config
            )
            await self.embedding_manager.initialize(self.embedding_models)
            
            # Initialize search engine
            self.search_engine = SemanticSearchEngine(
                self.embedding_manager,
                self.neo4j_uri, self.neo4j_user, self.neo4j_password,
                self.redis_config['host'], self.redis_config['port'], self.redis_config['password'],
                self.postgres_config
            )
            await self.search_engine.initialize(rebuild_indices)
            
            # Initialize graph retriever
            self.graph_retriever = GraphEnhancedRAGRetriever(
                self.search_engine,
                self.neo4j_uri, self.neo4j_user, self.neo4j_password,
                self.redis_config['host'], self.redis_config['port'], self.redis_config['password']
            )
            
            # Initialize educational components
            self.profile_manager = StudentProfileManager(self.postgres_pool, self.redis_client)
            self.content_analyzer = ContentAnalyzer(
                GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
            )
            self.context_ranker = ContextAwareRanker(self.profile_manager, self.content_analyzer)
            
            # Setup database schemas
            await setup_student_profile_tables(self.postgres_pool)
            
            initialization_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ RAG Pipeline initialized in {initialization_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG pipeline: {e}")
            raise
    
    async def close(self):
        """Close all database connections and cleanup"""
        logger.info("🔒 Closing RAG Pipeline connections")
        
        try:
            if self.embedding_manager:
                await self.embedding_manager.close()
            if self.search_engine:
                await self.search_engine.close()
            if self.graph_retriever:
                await self.graph_retriever.close()
            if self.postgres_pool:
                await self.postgres_pool.close()
            if self.redis_client:
                await asyncio.to_thread(self.redis_client.close)
                
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
    async def _initialize_databases(self):
        """Initialize database connections"""
        
        # PostgreSQL
        self.postgres_pool = await asyncpg.create_pool(**self.postgres_config)
        logger.info("✅ PostgreSQL connection pool initialized")
        
        # Redis
        self.redis_client = redis.Redis(
            host=self.redis_config['host'],
            port=self.redis_config['port'],
            password=self.redis_config['password'],
            decode_responses=True
        )
        # Test Redis connection
        await asyncio.to_thread(self.redis_client.ping)
        logger.info("✅ Redis client initialized")
    
    async def process_query(self, query: RAGQuery) -> RAGResponse:
        """Process a complete RAG query through the pipeline"""
        
        start_time = datetime.now()
        
        # Update query statistics
        self.query_stats['total_queries'] += 1
        
        try:
            # Check cache first if enabled
            if query.use_cache:
                cached_response = await self._get_cached_response(query)
                if cached_response:
                    cached_response.cache_hit = True
                    return cached_response
            
            # Create learning context
            context = await self._create_learning_context(query)
            
            # Execute RAG pipeline based on mode
            response = await self._execute_rag_mode(query, context)
            
            # Post-process response
            response = await self._post_process_response(response, query, context)
            
            # Cache response if enabled
            if query.use_cache:
                await self._cache_response(query, response)
            
            # Update performance statistics
            response.processing_time = (datetime.now() - start_time).total_seconds()
            await self._update_query_stats(response)
            
            logger.info(f"🎉 RAG query processed in {response.processing_time:.2f}s, {len(response.results)} results")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ RAG query processing failed: {e}")
            
            # Return error response
            error_response = RAGResponse(
                query=query.text,
                user_id=query.user_id,
                session_id=query.session_id,
                results=[],
                total_results=0,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            error_response.reasoning_explanation = f"Query processing failed: {str(e)}"
            
            return error_response
    
    async def _create_learning_context(self, query: RAGQuery) -> LearningContext:
        """Create learning context from query and user history"""
        
        # Get previous queries from session
        previous_queries = await self._get_session_queries(query.session_id)
        
        # Estimate current difficulty level and trends
        query_complexity = await self._analyze_query_complexity(query.text)
        complexity_trend = await self._analyze_complexity_trend(query.session_id)
        
        # Estimate focus level based on session patterns
        focus_level = await self._estimate_focus_level(query.user_id, query.session_id)
        
        context = LearningContext(
            session_id=query.session_id,
            user_id=query.user_id,
            current_topic=query.current_topic,
            learning_objective=query.learning_objective,
            previous_queries=previous_queries[-5:],  # Last 5 queries
            current_difficulty_level=query.student_level,
            query_complexity_trend=complexity_trend,
            estimated_focus_level=focus_level
        )
        
        return context
    
    async def _execute_rag_mode(self, query: RAGQuery, context: LearningContext) -> RAGResponse:
        """Execute RAG pipeline based on selected mode"""
        
        if query.mode == RAGMode.QUICK:
            return await self._execute_quick_mode(query, context)
        elif query.mode == RAGMode.COMPREHENSIVE:
            return await self._execute_comprehensive_mode(query, context)
        elif query.mode == RAGMode.EDUCATIONAL:
            return await self._execute_educational_mode(query, context)
        elif query.mode == RAGMode.RESEARCH:
            return await self._execute_research_mode(query, context)
        else:
            # Default to comprehensive mode
            return await self._execute_comprehensive_mode(query, context)
    
    async def _execute_quick_mode(self, query: RAGQuery, context: LearningContext) -> RAGResponse:
        """Execute quick RAG mode - semantic search only"""
        
        search_query = SearchQuery(
            text=query.text,
            search_type=SearchType.SEMANTIC_ONLY,
            content_types=query.content_types,
            limit=query.limit,
            min_similarity=query.min_similarity
        )
        
        results = await self.search_engine.search(search_query)
        
        return RAGResponse(
            query=query.text,
            user_id=query.user_id,
            session_id=query.session_id,
            results=results,
            total_results=len(results),
            strategies_used=['semantic_search']
        )
    
    async def _execute_comprehensive_mode(self, query: RAGQuery, context: LearningContext) -> RAGResponse:
        """Execute comprehensive RAG mode with graph enhancement"""
        
        search_query = SearchQuery(
            text=query.text,
            search_type=query.search_type,
            content_types=query.content_types,
            limit=query.limit * 2,  # Get more for filtering
            min_similarity=query.min_similarity,
            include_related=query.include_related
        )
        
        # Get enhanced results with graph traversal
        results = await self.graph_retriever.enhanced_retrieve(
            search_query,
            traversal_strategy=query.traversal_strategy,
            include_learning_paths=query.include_learning_paths,
            student_level=query.student_level
        )
        
        # Apply personalized ranking if enabled
        if query.use_personalization:
            results = await self.context_ranker.rank_results(
                results, query.user_id, context
            )
        
        # Limit to requested number
        results = results[:query.limit]
        
        return RAGResponse(
            query=query.text,
            user_id=query.user_id,
            session_id=query.session_id,
            results=results,
            total_results=len(results),
            strategies_used=['semantic_search', 'graph_traversal', 'personalized_ranking']
        )
    
    async def _execute_educational_mode(self, query: RAGQuery, context: LearningContext) -> RAGResponse:
        """Execute educational RAG mode optimized for learning"""
        
        # Start with comprehensive retrieval
        response = await self._execute_comprehensive_mode(query, context)
        
        # Add educational insights
        response.learning_insights = await self._generate_learning_insights(
            query, response.results, context
        )
        
        # Find learning paths if requested
        if query.include_learning_paths:
            response.learning_path = await self._generate_learning_paths(
                query, response.results, context
            )
        
        # Find related concepts
        if query.include_related_concepts:
            response.related_concepts = await self._find_related_concepts(
                query, response.results
            )
        
        # Identify prerequisite gaps
        response.prerequisite_gaps = await self._identify_prerequisite_gaps(
            query.user_id, response.results
        )
        
        # Generate next steps
        response.suggested_next_steps = await self._generate_next_steps(
            query, response.results, context
        )
        
        response.strategies_used.extend(['educational_analysis', 'learning_path_generation'])
        
        return response
    
    async def _execute_research_mode(self, query: RAGQuery, context: LearningContext) -> RAGResponse:
        """Execute research RAG mode with multiple strategies"""
        
        # Run multiple retrieval strategies in parallel
        strategies = [
            (GraphTraversalStrategy.RANDOM_WALK, "random_walk"),
            (GraphTraversalStrategy.PERSONALIZED_PAGERANK, "pagerank"),
            (GraphTraversalStrategy.CONCEPT_HIERARCHY, "hierarchy")
        ]
        
        all_results = []
        used_strategies = ['multi_strategy_retrieval']
        
        # Execute different strategies
        for strategy, strategy_name in strategies:
            search_query = SearchQuery(
                text=query.text,
                search_type=SearchType.HYBRID,
                content_types=query.content_types,
                limit=query.limit,
                min_similarity=query.min_similarity * 0.8  # Lower threshold for research
            )
            
            strategy_results = await self.graph_retriever.enhanced_retrieve(
                search_query,
                traversal_strategy=strategy,
                student_level=query.student_level
            )
            
            # Tag results with strategy
            for result in strategy_results:
                result.metadata['discovery_strategy'] = strategy_name
            
            all_results.extend(strategy_results)
            used_strategies.append(strategy_name)
        
        # Combine and deduplicate results
        unique_results = self._deduplicate_results(all_results)
        
        # Apply advanced ranking
        if query.use_personalization:
            unique_results = await self.context_ranker.rank_results(
                unique_results, query.user_id, context
            )
        
        # Limit results
        final_results = unique_results[:query.limit]
        
        return RAGResponse(
            query=query.text,
            user_id=query.user_id,
            session_id=query.session_id,
            results=final_results,
            total_results=len(final_results),
            strategies_used=used_strategies
        )
    
    async def _post_process_response(self, response: RAGResponse, 
                                   query: RAGQuery, context: LearningContext) -> RAGResponse:
        """Post-process response with quality metrics and formatting"""
        
        if not response.results:
            return response
        
        # Calculate quality metrics
        response.result_quality_score = self._calculate_quality_score(response.results)
        response.coverage_score = await self._calculate_coverage_score(query, response.results)
        response.confidence_score = self._calculate_confidence_score(response.results)
        
        # Apply personalization insights
        if query.use_personalization:
            response.personalization_applied = await self._get_personalization_summary(
                query.user_id, response.results
            )
        
        # Format for LLM consumption if requested
        if query.format_for_llm:
            response.formatted_context = self._format_for_llm(response.results, query)
            response.reasoning_explanation = self._generate_reasoning_explanation(
                query, response, context
            )
        
        return response
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on node_id"""
        
        seen_nodes = set()
        unique_results = []
        
        for result in results:
            if result.node_id not in seen_nodes:
                seen_nodes.add(result.node_id)
                unique_results.append(result)
        
        # Sort by similarity score
        unique_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(unique_results):
            result.rank = i + 1
        
        return unique_results
    
    def _calculate_quality_score(self, results: List[SearchResult]) -> float:
        """Calculate overall quality score for results"""
        
        if not results:
            return 0.0
        
        # Average similarity score
        avg_similarity = sum(r.similarity_score for r in results) / len(results)
        
        # Content type diversity
        content_types = set(r.content_type for r in results)
        diversity_score = min(1.0, len(content_types) / 4)  # Max 4 content types
        
        # Quality = 70% similarity + 30% diversity
        return (avg_similarity * 0.7) + (diversity_score * 0.3)
    
    async def _calculate_coverage_score(self, query: RAGQuery, results: List[SearchResult]) -> float:
        """Calculate how well results cover the query"""
        
        if not results:
            return 0.0
        
        # Simple coverage based on result count and similarity
        result_coverage = min(1.0, len(results) / query.limit)
        similarity_coverage = sum(r.similarity_score for r in results[:5]) / 5  # Top 5 average
        
        return (result_coverage * 0.4) + (similarity_coverage * 0.6)
    
    def _calculate_confidence_score(self, results: List[SearchResult]) -> float:
        """Calculate confidence in the results"""
        
        if not results:
            return 0.0
        
        # Confidence based on top result similarity and result distribution
        top_similarity = results[0].similarity_score if results else 0.0
        
        # Check for consistent high-quality results
        high_quality_count = sum(1 for r in results[:5] if r.similarity_score > 0.7)
        consistency_score = high_quality_count / min(5, len(results))
        
        return (top_similarity * 0.6) + (consistency_score * 0.4)
    
    async def _generate_learning_insights(self, query: RAGQuery, results: List[SearchResult],
                                        context: LearningContext) -> Dict[str, Any]:
        """Generate educational insights from results"""
        
        insights = {
            'difficulty_analysis': {},
            'concept_coverage': {},
            'learning_recommendations': []
        }
        
        if not results:
            return insights
        
        # Analyze difficulty distribution
        difficulties = []
        for result in results:
            if 'difficulty_level' in result.metadata:
                diff = result.metadata['difficulty_level']
                if diff == 'beginner':
                    difficulties.append(1)
                elif diff == 'intermediate':
                    difficulties.append(2)
                elif diff == 'advanced':
                    difficulties.append(3)
        
        if difficulties:
            avg_difficulty = sum(difficulties) / len(difficulties)
            insights['difficulty_analysis'] = {
                'average_difficulty': avg_difficulty,
                'difficulty_range': max(difficulties) - min(difficulties),
                'recommendation': self._get_difficulty_recommendation(avg_difficulty, query.student_level)
            }
        
        # Analyze concept coverage
        concepts_covered = set()
        for result in results:
            # Extract concepts from content (simplified)
            content_words = result.content.lower().split()
            physics_concepts = ['velocity', 'acceleration', 'force', 'energy', 'momentum', 
                              'work', 'power', 'mass', 'displacement', 'friction']
            
            for concept in physics_concepts:
                if concept in content_words:
                    concepts_covered.add(concept)
        
        insights['concept_coverage'] = {
            'concepts_found': list(concepts_covered),
            'coverage_breadth': len(concepts_covered),
            'topic_focus': query.current_topic if query.current_topic else "general"
        }
        
        # Learning recommendations
        insights['learning_recommendations'] = await self._generate_learning_recommendations(
            query, results, context
        )
        
        return insights
    
    def _get_difficulty_recommendation(self, avg_difficulty: float, student_level: str) -> str:
        """Get difficulty recommendation based on analysis"""
        
        level_mapping = {'beginner': 1, 'intermediate': 2, 'advanced': 3}
        target_difficulty = level_mapping.get(student_level, 2)
        
        if avg_difficulty < target_difficulty - 0.5:
            return "Content may be too easy. Consider more challenging material."
        elif avg_difficulty > target_difficulty + 0.5:
            return "Content may be challenging. Consider reviewing prerequisites."
        else:
            return "Content difficulty is well-matched to your level."
    
    async def _generate_learning_recommendations(self, query: RAGQuery, results: List[SearchResult],
                                               context: LearningContext) -> List[str]:
        """Generate specific learning recommendations"""
        
        recommendations = []
        
        if not results:
            recommendations.append("Try rephrasing your question or using different keywords.")
            return recommendations
        
        # Content type recommendations
        content_types = [r.content_type for r in results[:5]]
        type_counts = {ct: content_types.count(ct) for ct in set(content_types)}
        
        if type_counts.get('concept', 0) > type_counts.get('problem', 0):
            recommendations.append("Practice with problems to reinforce conceptual understanding.")
        elif type_counts.get('problem', 0) > type_counts.get('explanation', 0):
            recommendations.append("Review conceptual explanations to deepen understanding.")
        
        # Difficulty progression recommendations
        if len(results) >= 3:
            difficulties = []
            for result in results[:3]:
                if 'difficulty_level' in result.metadata:
                    diff = result.metadata['difficulty_level']
                    if diff == 'beginner':
                        difficulties.append(1)
                    elif diff == 'intermediate':
                        difficulties.append(2)
                    elif diff == 'advanced':
                        difficulties.append(3)
            
            if difficulties:
                if max(difficulties) - min(difficulties) > 1:
                    recommendations.append("Consider starting with easier concepts before tackling advanced material.")
        
        # Session-based recommendations
        if context.session_duration > 30:
            recommendations.append("Take a break to maintain focus and retention.")
        
        if len(context.previous_queries) > 3:
            recommendations.append("You're exploring deeply - consider summarizing key concepts learned.")
        
        return recommendations
    
    async def _generate_learning_paths(self, query: RAGQuery, results: List[SearchResult],
                                     context: LearningContext) -> List[Dict]:
        """Generate learning path suggestions"""
        
        if not self.graph_retriever or not results:
            return []
        
        # Use graph pathfinder to find learning paths
        pathfinder = self.graph_retriever.pathfinder
        
        # Extract key concepts from top results
        key_concepts = []
        for result in results[:3]:
            if result.content_type == 'concept':
                key_concepts.append(result.title.lower())
        
        if len(key_concepts) < 2:
            return []
        
        # Find learning path between concepts
        try:
            paths = await pathfinder.find_learning_path(
                key_concepts[0], key_concepts[1], query.student_level
            )
            
            learning_paths = []
            for path in paths[:2]:  # Top 2 paths
                path_dict = {
                    'concepts': [node.name for node in path.nodes],
                    'explanation': path.explanation,
                    'difficulty_progression': self._analyze_path_difficulty(path.nodes),
                    'estimated_time': len(path.nodes) * 10  # 10 minutes per concept
                }
                learning_paths.append(path_dict)
            
            return learning_paths
            
        except Exception as e:
            logger.warning(f"Failed to generate learning paths: {e}")
            return []
    
    def _analyze_path_difficulty(self, nodes) -> str:
        """Analyze difficulty progression in a learning path"""
        
        difficulties = []
        for node in nodes:
            if hasattr(node, 'properties') and 'difficulty_level' in node.properties:
                diff = node.properties['difficulty_level']
                if diff == 'beginner':
                    difficulties.append(1)
                elif diff == 'intermediate':
                    difficulties.append(2)
                elif diff == 'advanced':
                    difficulties.append(3)
        
        if not difficulties:
            return "unknown"
        
        # Analyze progression
        increasing = all(difficulties[i] <= difficulties[i+1] for i in range(len(difficulties)-1))
        decreasing = all(difficulties[i] >= difficulties[i+1] for i in range(len(difficulties)-1))
        
        if increasing:
            return "gradual_increase"
        elif decreasing:
            return "gradual_decrease"
        else:
            return "mixed"
    
    async def _find_related_concepts(self, query: RAGQuery, results: List[SearchResult]) -> List[Dict]:
        """Find concepts related to the query results"""
        
        if not self.graph_retriever or not results:
            return []
        
        pathfinder = self.graph_retriever.pathfinder
        related_concepts = []
        
        # Get related concepts for top results
        for result in results[:2]:  # Top 2 results
            if result.content_type == 'concept':
                try:
                    related_nodes = await pathfinder.find_related_concepts(
                        result.title, max_hops=2
                    )
                    
                    for node in related_nodes[:3]:  # Top 3 related per concept
                        concept_dict = {
                            'name': node.name,
                            'description': node.description,
                            'relationship_type': node.relationships[0]['type'] if node.relationships else 'related',
                            'source_concept': result.title
                        }
                        related_concepts.append(concept_dict)
                        
                except Exception as e:
                    logger.warning(f"Failed to find related concepts: {e}")
        
        return related_concepts
    
    async def _identify_prerequisite_gaps(self, user_id: str, results: List[SearchResult]) -> List[str]:
        """Identify missing prerequisites for the user"""
        
        if not self.profile_manager:
            return []
        
        try:
            profile = await self.profile_manager.get_student_profile(user_id)
            if not profile:
                return []
            
            gaps = []
            
            # Check for concepts in results that user is struggling with
            for result in results:
                content_lower = result.content.lower()
                
                for struggling_concept in profile.struggling_concepts:
                    if struggling_concept.lower() in content_lower:
                        gaps.append(f"Review {struggling_concept} - identified as challenging")
            
            # Check for concepts with low understanding
            for concept, understanding in profile.concept_understanding.items():
                if understanding < 0.4:  # Low understanding threshold
                    for result in results:
                        if concept.lower() in result.content.lower():
                            gaps.append(f"Strengthen understanding of {concept}")
                            break
            
            return gaps[:5]  # Limit to top 5 gaps
            
        except Exception as e:
            logger.warning(f"Failed to identify prerequisite gaps: {e}")
            return []
    
    async def _generate_next_steps(self, query: RAGQuery, results: List[SearchResult],
                                 context: LearningContext) -> List[str]:
        """Generate suggested next steps for learning"""
        
        next_steps = []
        
        if not results:
            next_steps.append("Try searching with different keywords or ask a more specific question.")
            return next_steps
        
        # Based on content types found
        content_types = [r.content_type for r in results[:5]]
        
        if 'concept' in content_types and 'problem' not in content_types:
            next_steps.append("Practice with problems to apply these concepts.")
        
        if 'formula' in content_types:
            next_steps.append("Work through examples using these formulas.")
        
        if 'explanation' in content_types:
            next_steps.append("Test your understanding with practice questions.")
        
        # Based on difficulty analysis
        avg_score = sum(r.similarity_score for r in results[:3]) / min(3, len(results))
        
        if avg_score > 0.8:
            next_steps.append("You have good resources - focus on active practice and application.")
        elif avg_score < 0.5:
            next_steps.append("Consider refining your question or exploring related fundamental concepts first.")
        
        # Based on session context
        if context.session_duration > 45:
            next_steps.append("Consider taking a break and reviewing what you've learned so far.")
        
        return next_steps[:4]  # Limit to top 4 suggestions
    
    def _format_for_llm(self, results: List[SearchResult], query: RAGQuery) -> str:
        """Format results for LLM consumption"""
        
        if not results:
            return "No relevant information found for the query."
        
        formatted_parts = []
        formatted_parts.append(f"Query: {query.text}\n")
        formatted_parts.append("Relevant Information:\n")
        
        for i, result in enumerate(results[:5], 1):  # Top 5 results for LLM
            formatted_parts.append(f"{i}. {result.title}")
            formatted_parts.append(f"   Type: {result.content_type}")
            formatted_parts.append(f"   Content: {result.content[:200]}...")  # Truncate for readability
            formatted_parts.append(f"   Relevance: {result.similarity_score:.2f}")
            
            if result.explanation:
                formatted_parts.append(f"   Context: {result.explanation}")
            
            formatted_parts.append("")  # Empty line for readability
        
        return "\n".join(formatted_parts)
    
    def _generate_reasoning_explanation(self, query: RAGQuery, response: RAGResponse,
                                       context: LearningContext) -> str:
        """Generate explanation of the reasoning behind the results"""
        
        explanation_parts = []
        
        explanation_parts.append(f"Search Strategy: Used {', '.join(response.strategies_used)} to find relevant content.")
        
        if response.results:
            explanation_parts.append(f"Found {len(response.results)} relevant results with average relevance of {response.result_quality_score:.2f}.")
            
            # Content type distribution
            content_types = [r.content_type for r in response.results]
            type_counts = {ct: content_types.count(ct) for ct in set(content_types)}
            explanation_parts.append(f"Content distribution: {', '.join(f'{ct}: {count}' for ct, count in type_counts.items())}")
        
        if response.personalization_applied:
            explanation_parts.append("Results were personalized based on your learning profile and preferences.")
        
        if response.learning_insights:
            insights = response.learning_insights
            if 'difficulty_analysis' in insights and 'recommendation' in insights['difficulty_analysis']:
                explanation_parts.append(f"Difficulty assessment: {insights['difficulty_analysis']['recommendation']}")
        
        return " ".join(explanation_parts)
    
    async def _get_personalization_summary(self, user_id: str, results: List[SearchResult]) -> Dict[str, Any]:
        """Get summary of personalization applied"""
        
        if not self.profile_manager:
            return {}
        
        try:
            profile = await self.profile_manager.get_student_profile(user_id)
            if not profile:
                return {}
            
            personalization_summary = {
                'student_level': profile.current_level,
                'learning_style': profile.learning_style.value,
                'content_preferences': [p.value for p in profile.content_preferences],
                'personalization_factors': []
            }
            
            # Check which personalization factors were applied
            for result in results[:3]:
                if 'personalization_factors' in result.metadata:
                    personalization_summary['personalization_factors'].extend(
                        list(result.metadata['personalization_factors'].values())
                    )
            
            return personalization_summary
            
        except Exception as e:
            logger.warning(f"Failed to get personalization summary: {e}")
            return {}
    
    # Caching methods
    
    async def _get_cached_response(self, query: RAGQuery) -> Optional[RAGResponse]:
        """Get cached response if available"""
        
        cache_key = self._generate_cache_key(query)
        
        try:
            cached_data = await asyncio.to_thread(self.redis_client.get, cache_key)
            if cached_data:
                response_dict = json.loads(cached_data)
                # Note: In production, you'd want proper serialization/deserialization
                # This is a simplified version
                return None  # Skip caching for now
                
        except Exception as e:
            logger.warning(f"Failed to get cached response: {e}")
        
        return None
    
    async def _cache_response(self, query: RAGQuery, response: RAGResponse):
        """Cache response for future queries"""
        
        if not query.use_cache:
            return
        
        cache_key = self._generate_cache_key(query)
        
        try:
            # Simplified caching - in production, use proper serialization
            cache_data = {
                'query': query.text,
                'user_id': query.user_id,
                'timestamp': datetime.now().isoformat(),
                'result_count': len(response.results)
            }
            
            await asyncio.to_thread(
                self.redis_client.setex,
                cache_key,
                query.cache_ttl,
                json.dumps(cache_data)
            )
            
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")
    
    def _generate_cache_key(self, query: RAGQuery) -> str:
        """Generate cache key for query"""
        
        # Create hash from query parameters
        cache_components = [
            query.text.lower(),
            query.mode.value,
            query.search_type.value,
            str(query.limit),
            str(sorted(query.content_types)) if query.content_types else 'all'
        ]
        
        cache_string = "|".join(cache_components)
        import hashlib
        cache_hash = hashlib.md5(cache_string.encode()).hexdigest()
        
        return f"rag_response:{cache_hash}"
    
    # Utility methods for session analysis
    
    async def _get_session_queries(self, session_id: str) -> List[str]:
        """Get previous queries from session"""
        
        if not session_id:
            return []
        
        try:
            session_key = f"session_queries:{session_id}"
            queries_data = await asyncio.to_thread(self.redis_client.lrange, session_key, 0, -1)
            return queries_data if queries_data else []
            
        except Exception as e:
            logger.warning(f"Failed to get session queries: {e}")
            return []
    
    async def _analyze_query_complexity(self, query_text: str) -> float:
        """Analyze complexity of a query"""
        
        complexity_indicators = [
            'derive', 'proof', 'explain why', 'how does', 'relationship between',
            'compare', 'analyze', 'calculate', 'solve', 'determine'
        ]
        
        query_lower = query_text.lower()
        complexity_score = sum(1 for indicator in complexity_indicators if indicator in query_lower)
        
        # Normalize to 0-1 scale
        return min(1.0, complexity_score * 0.2)
    
    async def _analyze_complexity_trend(self, session_id: str) -> str:
        """Analyze complexity trend in session"""
        
        queries = await self._get_session_queries(session_id)
        
        if len(queries) < 3:
            return "stable"
        
        # Analyze last 3 queries
        complexities = []
        for query in queries[-3:]:
            complexity = await self._analyze_query_complexity(query)
            complexities.append(complexity)
        
        # Simple trend analysis
        if complexities[-1] > complexities[0] + 0.2:
            return "increasing"
        elif complexities[-1] < complexities[0] - 0.2:
            return "decreasing"
        else:
            return "stable"
    
    async def _estimate_focus_level(self, user_id: str, session_id: str) -> float:
        """Estimate current focus level"""
        
        # Simplified focus estimation
        base_focus = 1.0
        
        # Get session duration
        try:
            session_key = f"session:{session_id}"
            session_data = await asyncio.to_thread(self.redis_client.get, session_key)
            
            if session_data:
                session_info = json.loads(session_data)
                start_time = datetime.fromisoformat(session_info.get('start_time', datetime.now().isoformat()))
                session_duration = (datetime.now() - start_time).total_seconds() / 60  # minutes
                
                # Focus decreases after 30 minutes
                if session_duration > 30:
                    base_focus -= (session_duration - 30) * 0.01
                
        except Exception as e:
            logger.warning(f"Failed to estimate focus level: {e}")
        
        return max(0.3, min(1.0, base_focus))
    
    async def _update_query_stats(self, response: RAGResponse):
        """Update query performance statistics"""
        
        # Update success rate
        success = 1.0 if response.results else 0.0
        current_success = self.query_stats['success_rate']
        total_queries = self.query_stats['total_queries']
        
        self.query_stats['success_rate'] = ((current_success * (total_queries - 1)) + success) / total_queries
        
        # Update average response time
        current_avg_time = self.query_stats['avg_response_time']
        self.query_stats['avg_response_time'] = ((current_avg_time * (total_queries - 1)) + response.processing_time) / total_queries
        
        # Update cache hit rate
        cache_hit = 1.0 if response.cache_hit else 0.0
        current_cache_rate = self.query_stats['cache_hit_rate']
        self.query_stats['cache_hit_rate'] = ((current_cache_rate * (total_queries - 1)) + cache_hit) / total_queries

# Context manager for easy usage
@asynccontextmanager
async def get_rag_pipeline(
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "physics_graph_password_2024",
    postgres_config: dict = None,
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_password: str = "redis_secure_password_2024",
    embedding_models: List[str] = None,
    rebuild_indices: bool = False
):
    """Context manager for RAG pipeline"""
    
    if postgres_config is None:
        postgres_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'physics_assistant',
            'user': 'physics_user',
            'password': 'physics_secure_password_2024'
        }
    
    pipeline = RAGPipeline(
        neo4j_uri, neo4j_user, neo4j_password,
        postgres_config, redis_host, redis_port, redis_password,
        embedding_models
    )
    
    try:
        await pipeline.initialize(rebuild_indices)
        yield pipeline
    finally:
        await pipeline.close()

# Example usage and testing
async def test_rag_pipeline():
    """Test the complete RAG pipeline"""
    
    async with get_rag_pipeline() as pipeline:
        
        # Test different RAG modes
        test_queries = [
            RAGQuery(
                text="What is Newton's second law?",
                user_id="test_user_123",
                session_id="test_session_1",
                mode=RAGMode.QUICK,
                limit=5
            ),
            RAGQuery(
                text="How does energy conservation apply to collisions?",
                user_id="test_user_123",
                session_id="test_session_1",
                mode=RAGMode.COMPREHENSIVE,
                student_level="intermediate",
                current_topic="momentum",
                limit=8
            ),
            RAGQuery(
                text="Explain the relationship between force and acceleration",
                user_id="test_user_123",
                session_id="test_session_1",
                mode=RAGMode.EDUCATIONAL,
                include_learning_paths=True,
                include_related_concepts=True,
                limit=10
            )
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*50}")
            print(f"TEST {i}: {query.mode.value.upper()} MODE")
            print(f"Query: {query.text}")
            print(f"{'='*50}")
            
            response = await pipeline.process_query(query)
            
            print(f"📊 Results: {len(response.results)} found")
            print(f"⏱️  Processing time: {response.processing_time:.2f}s")
            print(f"🎯 Quality score: {response.result_quality_score:.2f}")
            print(f"📋 Strategies: {', '.join(response.strategies_used)}")
            
            print("\nTop Results:")
            for result in response.results[:3]:
                print(f"  {result.rank}. {result.title}")
                print(f"     Type: {result.content_type}, Score: {result.similarity_score:.3f}")
                if result.explanation:
                    print(f"     Context: {result.explanation}")
            
            if response.learning_insights:
                print(f"\n🎓 Learning Insights:")
                insights = response.learning_insights
                if 'difficulty_analysis' in insights:
                    print(f"  Difficulty: {insights['difficulty_analysis'].get('recommendation', 'N/A')}")
                if 'learning_recommendations' in insights:
                    for rec in insights['learning_recommendations'][:2]:
                        print(f"  • {rec}")
            
            if response.suggested_next_steps:
                print(f"\n➡️  Next Steps:")
                for step in response.suggested_next_steps[:2]:
                    print(f"  • {step}")
            
            if response.formatted_context and len(response.formatted_context) > 0:
                print(f"\n🤖 LLM Context Preview:")
                print(response.formatted_context[:200] + "...")
        
        print(f"\n{'='*50}")
        print("PIPELINE STATISTICS")
        print(f"{'='*50}")
        stats = pipeline.query_stats
        print(f"Total queries: {stats['total_queries']}")
        print(f"Success rate: {stats['success_rate']:.2%}")
        print(f"Average response time: {stats['avg_response_time']:.2f}s")
        print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")

if __name__ == "__main__":
    asyncio.run(test_rag_pipeline())