"""
Physics Assistant RAG System
Phase 3.3: Semantic Search and Retrieval System

This module implements a comprehensive RAG (Retrieval-Augmented Generation) system
specifically designed for physics education. It combines vector embeddings, semantic search,
graph traversal, and context-aware ranking to provide intelligent content retrieval.

Components:
- vector_embeddings.py: Physics-optimized embedding generation system
- semantic_search.py: Hybrid semantic and keyword search engine
- graph_retrieval.py: Graph-enhanced retrieval with educational pathfinding
- context_aware_ranking.py: Personalized ranking based on student profiles
- rag_pipeline.py: Complete RAG query processing pipeline
- performance_optimization.py: Advanced caching and indexing optimizations

Key Features:
- Multi-modal embeddings for physics concepts, formulas, problems, and explanations
- Hybrid search combining vector similarity with keyword matching
- Graph traversal algorithms for educational content discovery
- Student profile-aware personalization and adaptive difficulty
- Learning path generation and prerequisite analysis
- Performance optimization with multi-level caching and FAISS indexing
- Comprehensive monitoring and analytics
"""

__version__ = "1.0.0"
__author__ = "Physics Assistant RAG Team"

# Import main classes and functions for easy access
from .vector_embeddings import (
    PhysicsEmbeddingManager,
    EmbeddingConfig,
    get_physics_embedding_manager
)

from .semantic_search import (
    SemanticSearchEngine,
    SearchQuery,
    SearchResult,
    SearchType,
    PhysicsQueryProcessor
)

from .graph_retrieval import (
    GraphEnhancedRAGRetriever,
    EducationalPathfinder,
    GraphTraversalStrategy,
    GraphNode,
    GraphPath
)

from .context_aware_ranking import (
    ContextAwareRanker,
    StudentProfileManager,
    ContentAnalyzer,
    StudentProfile,
    LearningContext,
    LearningStyle,
    ContentPreference
)

from .rag_pipeline import (
    RAGPipeline,
    RAGQuery,
    RAGResponse,
    RAGMode,
    get_rag_pipeline
)

from .performance_optimization import (
    AdvancedCache,
    OptimizedIndexManager,
    PerformanceMonitor,
    CacheConfig,
    IndexConfig,
    PerformanceMetrics,
    create_optimized_rag_components
)

# Module metadata
__all__ = [
    # Core classes
    "RAGPipeline",
    "RAGQuery", 
    "RAGResponse",
    "RAGMode",
    
    # Embedding system
    "PhysicsEmbeddingManager",
    "EmbeddingConfig",
    
    # Search system
    "SemanticSearchEngine",
    "SearchQuery",
    "SearchResult", 
    "SearchType",
    "PhysicsQueryProcessor",
    
    # Graph retrieval
    "GraphEnhancedRAGRetriever",
    "EducationalPathfinder",
    "GraphTraversalStrategy",
    "GraphNode",
    "GraphPath",
    
    # Context-aware ranking
    "ContextAwareRanker",
    "StudentProfileManager",
    "ContentAnalyzer", 
    "StudentProfile",
    "LearningContext",
    "LearningStyle",
    "ContentPreference",
    
    # Performance optimization
    "AdvancedCache",
    "OptimizedIndexManager",
    "PerformanceMonitor",
    "CacheConfig",
    "IndexConfig",
    "PerformanceMetrics",
    
    # Factory functions
    "get_physics_embedding_manager",
    "get_rag_pipeline",
    "create_optimized_rag_components"
]

# System configuration defaults
DEFAULT_CONFIG = {
    "embedding": {
        "default_model": "sentence_transformer",
        "available_models": [
            "sentence_transformer",
            "physics_optimized", 
            "openai_model",
            "math_specialized"
        ],
        "cache_ttl": 7200  # 2 hours
    },
    
    "search": {
        "default_limit": 10,
        "min_similarity": 0.3,
        "hybrid_semantic_weight": 0.7,
        "hybrid_keyword_weight": 0.3
    },
    
    "graph": {
        "max_walk_length": 5,
        "max_related_concepts": 10,
        "prerequisite_depth": 3
    },
    
    "personalization": {
        "profile_cache_ttl": 3600,  # 1 hour
        "difficulty_adaptation": True,
        "learning_style_weight": 0.2
    },
    
    "performance": {
        "enable_caching": True,
        "cache_compression": True,
        "async_cache_writes": True,
        "index_backup_interval": 3600
    }
}

# Validation functions
def validate_config(config: dict) -> bool:
    """Validate RAG system configuration"""
    required_keys = ["embedding", "search", "graph", "personalization", "performance"]
    return all(key in config for key in required_keys)

def get_default_config() -> dict:
    """Get default system configuration"""
    return DEFAULT_CONFIG.copy()

# System status and health check
async def check_system_health() -> dict:
    """Check health status of all RAG system components"""
    
    health_status = {
        "overall_status": "unknown",
        "components": {
            "embedding_system": "unknown",
            "search_engine": "unknown", 
            "graph_retrieval": "unknown",
            "ranking_system": "unknown",
            "performance_optimization": "unknown"
        },
        "metrics": {
            "total_embeddings": 0,
            "search_indices_built": False,
            "cache_operational": False,
            "monitoring_active": False
        },
        "last_check": None
    }
    
    # This would be implemented to check actual system health
    # For now, return basic structure
    
    return health_status

# Module initialization check
def is_system_ready() -> bool:
    """Check if the RAG system is ready for queries"""
    
    # This would check if all components are initialized
    # For now, return False as placeholder
    return False

# Usage examples and documentation
USAGE_EXAMPLES = {
    "basic_query": """
    from database.rag_system import get_rag_pipeline, RAGQuery, RAGMode
    
    async with get_rag_pipeline() as pipeline:
        query = RAGQuery(
            text="What is Newton's second law?",
            user_id="student_123",
            mode=RAGMode.EDUCATIONAL,
            student_level="intermediate"
        )
        
        response = await pipeline.process_query(query)
        print(f"Found {len(response.results)} results")
    """,
    
    "embedding_generation": """
    from database.rag_system import get_physics_embedding_manager
    
    async with get_physics_embedding_manager() as manager:
        embeddings_data = await manager.generate_content_embeddings()
        await manager.store_embeddings_in_neo4j(embeddings_data)
    """,
    
    "performance_optimization": """
    from database.rag_system import create_optimized_rag_components, CacheConfig
    
    cache_config = CacheConfig(
        redis_host="localhost",
        embedding_cache_ttl=7200,
        cache_compression=True
    )
    
    cache, index_manager, monitor = await create_optimized_rag_components(cache_config)
    """
}

# System information
def get_system_info() -> dict:
    """Get comprehensive system information"""
    
    return {
        "version": __version__,
        "author": __author__,
        "components": list(__all__),
        "config_defaults": DEFAULT_CONFIG,
        "usage_examples": USAGE_EXAMPLES,
        "documentation": __doc__
    }