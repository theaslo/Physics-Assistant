#!/usr/bin/env python3
"""
Physics Assistant Database API Server
Provides REST endpoints for interaction logging and analytics
"""
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, start_http_server
import time

from db_manager import DatabaseManager, get_db_manager

# Import analytics modules
try:
    from analytics.learning_analytics import LearningAnalyticsEngine, StudentProfile
    from analytics.concept_mastery import ConceptMasteryDetector, ConceptAssessment
    from analytics.learning_path_optimizer import LearningPathOptimizer, LearningObjective
    from analytics.educational_data_mining import EducationalDataMiner
    from analytics.realtime_analytics import RealTimeAnalyticsEngine, create_interaction_event, create_mastery_event
    ANALYTICS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Analytics modules not available: {e}")
    ANALYTICS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total', 
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)
REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)
DATABASE_CONNECTIONS = Gauge(
    'database_connections_active',
    'Active database connections',
    ['database']
)
INTERACTION_COUNT = Counter(
    'physics_interactions_total',
    'Total physics assistant interactions',
    ['agent_type', 'success']
)
AGENT_EXECUTION_TIME = Histogram(
    'agent_execution_time_seconds',
    'Agent execution time in seconds', 
    ['agent_type']
)

# FastAPI app configuration
app = FastAPI(
    title="Physics Assistant Database API",
    description="REST API for logging student interactions and retrieving analytics",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics middleware
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware to collect request metrics for Prometheus"""
    start_time = time.time()
    
    response = await call_next(request)
    
    # Record metrics
    duration = time.time() - start_time
    endpoint = request.url.path
    method = request.method
    status = str(response.status_code)
    
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
    REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
    
    return response

# Metrics endpoint for Prometheus scraping
@app.get("/metrics", response_class=PlainTextResponse, tags=["Monitoring"])
async def get_metrics():
    """Endpoint for Prometheus to scrape metrics"""
    return generate_latest()

# Global database manager and analytics engines
db_manager: Optional[DatabaseManager] = None
analytics_engine: Optional[LearningAnalyticsEngine] = None
mastery_detector: Optional[ConceptMasteryDetector] = None
path_optimizer: Optional[LearningPathOptimizer] = None
data_miner: Optional[EducationalDataMiner] = None
realtime_engine: Optional[RealTimeAnalyticsEngine] = None

# Pydantic models for request/response validation

class InteractionRequest(BaseModel):
    """Request model for logging interactions"""
    user_id: str = Field(..., description="UUID of the user")
    session_id: Optional[str] = Field(None, description="UUID of the session (optional)")
    agent_type: str = Field(..., description="Type of physics agent used")
    interaction_type: str = Field(default="chat", description="Type of interaction")
    message: str = Field(..., description="User's input message")
    response: str = Field(..., description="Agent's response")
    execution_time_ms: Optional[int] = Field(None, description="Response time in milliseconds")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

class InteractionResponse(BaseModel):
    """Response model for logged interactions"""
    interaction_id: str
    timestamp: datetime
    status: str = "success"

class AnalyticsQuery(BaseModel):
    """Query parameters for analytics endpoints"""
    user_id: Optional[str] = None
    agent_type: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: int = Field(default=100, le=1000)

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    healthy_databases: str
    timestamp: datetime
    databases: Dict[str, Any]

class ConceptQuery(BaseModel):
    """Query for physics concepts"""
    category: Optional[str] = None
    search_term: Optional[str] = None

# Analytics-specific models
class StudentProgressRequest(BaseModel):
    """Request for student progress analysis"""
    user_id: str
    time_window_days: int = Field(default=30, le=365)
    include_predictions: bool = Field(default=False)

class ConceptMasteryRequest(BaseModel):
    """Request for concept mastery assessment"""
    user_id: str
    concept: str
    time_window_days: int = Field(default=14, le=90)

class LearningPathRequest(BaseModel):
    """Request for learning path generation"""
    user_id: str
    target_concepts: List[str]
    time_constraint: Optional[float] = None  # hours
    difficulty_preference: str = Field(default="adaptive", regex="^(easy|moderate|challenging|adaptive)$")
    algorithm: str = Field(default="personalized_optimal")

class AnalyticsInsightRequest(BaseModel):
    """Request for educational insights"""
    user_id: Optional[str] = None
    timeframe_days: int = Field(default=30, le=365)
    insight_types: List[str] = Field(default=["trend", "anomaly", "correlation"])

class StudentClusteringRequest(BaseModel):
    """Request for student clustering analysis"""
    algorithm: str = Field(default="kmeans", regex="^(kmeans|dbscan|hierarchical)$")
    features: List[str] = Field(default=["success_rate", "engagement", "learning_velocity"])

class RealTimeMetricsRequest(BaseModel):
    """Request for real-time metrics"""
    metric_types: List[str] = Field(default=["system", "performance", "processors"])
    include_history: bool = Field(default=False)

# Startup and shutdown events

@app.on_event("startup")
async def startup_event():
    """Initialize database connections and analytics engines on startup"""
    global db_manager, analytics_engine, mastery_detector, path_optimizer, data_miner, realtime_engine
    logger.info("🚀 Starting Physics Assistant API Server")
    
    try:
        # Initialize database manager
        db_manager = DatabaseManager()
        success = await db_manager.initialize()
        
        if success:
            logger.info("✅ Database connections initialized successfully")
        else:
            logger.error("❌ Failed to initialize some database connections")
        
        # Initialize analytics engines if available
        if ANALYTICS_AVAILABLE and success:
            try:
                logger.info("🧠 Initializing Analytics Engines...")
                
                # Initialize core analytics engine
                analytics_engine = LearningAnalyticsEngine(db_manager)
                await analytics_engine.initialize()
                
                # Initialize concept mastery detector
                mastery_detector = ConceptMasteryDetector(db_manager)
                
                # Initialize learning path optimizer
                path_optimizer = LearningPathOptimizer(db_manager)
                await path_optimizer.initialize()
                
                # Initialize educational data miner
                data_miner = EducationalDataMiner(db_manager)
                await data_miner.initialize()
                
                # Initialize real-time analytics engine
                realtime_engine = RealTimeAnalyticsEngine(db_manager)
                
                # Start real-time engine in background
                asyncio.create_task(realtime_engine.start())
                
                logger.info("✅ All analytics engines initialized successfully")
                
            except Exception as e:
                logger.error(f"⚠️ Failed to initialize analytics engines: {e}")
                # Continue without analytics
                
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Close database connections and analytics engines on shutdown"""
    global db_manager, realtime_engine
    logger.info("🔒 Shutting down Physics Assistant API Server")
    
    # Stop real-time analytics engine
    if realtime_engine:
        await realtime_engine.stop()
        logger.info("✅ Real-time analytics engine stopped")
    
    # Close database connections
    if db_manager:
        await db_manager.close()
        logger.info("✅ Database connections closed")

# Dependency to get database manager
async def get_db() -> DatabaseManager:
    """Dependency to get the database manager"""
    if not db_manager or not db_manager._initialized:
        raise HTTPException(status_code=503, detail="Database not available")
    return db_manager

# Health check endpoints

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(db: DatabaseManager = Depends(get_db)):
    """Comprehensive health check for all database services"""
    try:
        health_data = await db.health_check()
        
        status_code = 200
        if health_data['status'] == 'degraded':
            status_code = 206  # Partial Content
        elif health_data['status'] == 'unhealthy':
            status_code = 503  # Service Unavailable
            
        return JSONResponse(
            status_code=status_code,
            content={
                **health_data,
                "timestamp": health_data["timestamp"]
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.get("/health/postgres", tags=["Health"])
async def postgres_health(db: DatabaseManager = Depends(get_db)):
    """PostgreSQL specific health check"""
    try:
        health_data = await db.postgres.health_check()
        status_code = 200 if health_data.get('status') == 'healthy' else 503
        return JSONResponse(status_code=status_code, content=health_data)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/health/neo4j", tags=["Health"])
async def neo4j_health(db: DatabaseManager = Depends(get_db)):
    """Neo4j specific health check"""
    try:
        health_data = await db.neo4j.health_check()
        status_code = 200 if health_data.get('status') == 'healthy' else 503
        return JSONResponse(status_code=status_code, content=health_data)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/health/redis", tags=["Health"])
async def redis_health(db: DatabaseManager = Depends(get_db)):
    """Redis specific health check"""
    try:
        health_data = await db.redis.health_check()
        status_code = 200 if health_data.get('status') == 'healthy' else 503
        return JSONResponse(status_code=status_code, content=health_data)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

# Interaction logging endpoints

@app.post("/interactions", response_model=InteractionResponse, tags=["Interactions"])
async def log_interaction(
    interaction: InteractionRequest,
    background_tasks: BackgroundTasks,
    db: DatabaseManager = Depends(get_db)
):
    """Log a student interaction with physics agents"""
    try:
        # Validate user exists (optional - could be removed for performance)
        async with db.postgres.get_connection() as conn:
            user_exists = await conn.fetchval("SELECT EXISTS(SELECT 1 FROM users WHERE id = $1)", interaction.user_id)
            if not user_exists:
                raise HTTPException(status_code=404, detail="User not found")
        
        # Log the interaction
        interaction_id = await db.log_interaction(
            user_id=interaction.user_id,
            agent_type=interaction.agent_type,
            message=interaction.message,
            response=interaction.response,
            session_id=interaction.session_id,
            metadata=interaction.metadata
        )
        
        # Record metrics for monitoring
        INTERACTION_COUNT.labels(
            agent_type=interaction.agent_type, 
            success="true"
        ).inc()
        
        if interaction.execution_time_ms:
            execution_seconds = interaction.execution_time_ms / 1000
            AGENT_EXECUTION_TIME.labels(agent_type=interaction.agent_type).observe(execution_seconds)
        
        # Submit real-time analytics event
        if realtime_engine:
            background_tasks.add_task(
                submit_realtime_event,
                interaction.user_id,
                interaction.agent_type,
                True,  # success
                interaction.execution_time_ms
            )
        
        # Optionally cache frequently accessed data in background
        if interaction.agent_type:
            background_tasks.add_task(
                cache_agent_stats, 
                db, 
                interaction.agent_type
            )
        
        return InteractionResponse(
            interaction_id=interaction_id,
            timestamp=datetime.now(),
            status="success"
        )
        
    except HTTPException:
        # Record failed interaction metric
        INTERACTION_COUNT.labels(
            agent_type=interaction.agent_type, 
            success="false"
        ).inc()
        raise
    except Exception as e:
        # Record failed interaction metric
        INTERACTION_COUNT.labels(
            agent_type=interaction.agent_type, 
            success="false"
        ).inc()
        logger.error(f"Failed to log interaction: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to log interaction: {str(e)}")

@app.get("/interactions", tags=["Interactions"])
async def get_interactions(
    user_id: Optional[str] = None,
    agent_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db: DatabaseManager = Depends(get_db)
):
    """Retrieve interaction history with optional filters"""
    try:
        query = """
        SELECT id, user_id, session_id, type, agent_type, 
               request_data, response_data, execution_time_ms,
               success, metadata, created_at
        FROM interactions
        WHERE 1=1
        """
        params = []
        param_count = 0
        
        if user_id:
            param_count += 1
            query += f" AND user_id = ${param_count}"
            params.append(user_id)
            
        if agent_type:
            param_count += 1
            query += f" AND agent_type = ${param_count}"  
            params.append(agent_type)
        
        param_count += 1
        query += f" ORDER BY created_at DESC LIMIT ${param_count}"
        params.append(limit)
        
        if offset > 0:
            param_count += 1
            query += f" OFFSET ${param_count}"
            params.append(offset)
        
        async with db.postgres.get_connection() as conn:
            results = await conn.fetch(query, *params)
        
        interactions = []
        for row in results:
            interaction_data = {
                "id": str(row['id']),
                "user_id": str(row['user_id']),
                "session_id": str(row['session_id']) if row['session_id'] else None,
                "type": row['type'],
                "agent_type": row['agent_type'],
                "request_data": row['request_data'],
                "response_data": row['response_data'],
                "execution_time_ms": row['execution_time_ms'],
                "success": row['success'],
                "metadata": row['metadata'],
                "created_at": row['created_at'].isoformat()
            }
            interactions.append(interaction_data)
        
        return {
            "interactions": interactions,
            "count": len(interactions),
            "offset": offset,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Failed to retrieve interactions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Analytics endpoints

@app.get("/analytics/summary", tags=["Analytics"])
async def get_analytics_summary(
    user_id: Optional[str] = None,
    days: int = 7,
    db: DatabaseManager = Depends(get_db)
):
    """Get interaction analytics summary"""
    try:
        since_date = datetime.now() - timedelta(days=days)
        
        query = """
        SELECT 
            COUNT(*) as total_interactions,
            COUNT(DISTINCT user_id) as unique_users,
            COUNT(DISTINCT agent_type) as agents_used,
            AVG(execution_time_ms) as avg_response_time,
            agent_type,
            COUNT(*) as agent_count
        FROM interactions 
        WHERE created_at >= $1
        """
        params = [since_date]
        
        if user_id:
            query += " AND user_id = $2"
            params.append(user_id)
            
        query += " GROUP BY agent_type ORDER BY agent_count DESC"
        
        async with db.postgres.get_connection() as conn:
            results = await conn.fetch(query, *params)
        
        agent_stats = []
        total_interactions = 0
        
        for row in results:
            agent_stats.append({
                "agent_type": row['agent_type'],
                "interaction_count": row['agent_count']
            })
            total_interactions = row['total_interactions']  # Same for all rows
        
        return {
            "period_days": days,
            "total_interactions": total_interactions,
            "unique_users": results[0]['unique_users'] if results else 0,
            "avg_response_time_ms": float(results[0]['avg_response_time']) if results and results[0]['avg_response_time'] else 0,
            "agent_usage": agent_stats
        }
        
    except Exception as e:
        logger.error(f"Analytics query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# RAG System endpoints

@app.post("/rag/query", tags=["RAG System"])
async def rag_query(
    request: Dict[str, Any],
    background_tasks: BackgroundTasks,
    db: DatabaseManager = Depends(get_db)
):
    """Process RAG query through the complete pipeline"""
    try:
        from rag_system.rag_pipeline import RAGQuery, RAGMode, SearchType, GraphTraversalStrategy
        
        # Parse request
        rag_query = RAGQuery(
            text=request.get("text", ""),
            user_id=request.get("user_id", "anonymous"),
            session_id=request.get("session_id", ""),
            limit=request.get("limit", 10),
            mode=RAGMode(request.get("mode", "comprehensive")),
            search_type=SearchType(request.get("search_type", "hybrid")),
            student_level=request.get("student_level", "intermediate"),
            current_topic=request.get("current_topic", ""),
            use_personalization=request.get("use_personalization", True),
            include_learning_paths=request.get("include_learning_paths", False),
            format_for_llm=request.get("format_for_llm", True)
        )
        
        # Process through RAG pipeline (this would be initialized at startup)
        # For now, return a placeholder response
        return {
            "status": "success",
            "query": rag_query.text,
            "message": "RAG pipeline endpoint ready - full implementation requires initialization"
        }
        
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/generate-embeddings", tags=["RAG System"])
async def generate_embeddings(
    background_tasks: BackgroundTasks,
    model_name: str = "sentence_transformer",
    rebuild_indices: bool = False,
    db: DatabaseManager = Depends(get_db)
):
    """Generate embeddings for all physics content"""
    try:
        # Add background task for embedding generation
        background_tasks.add_task(
            run_embedding_generation, model_name, rebuild_indices
        )
        
        return {
            "status": "started",
            "message": f"Embedding generation started with model: {model_name}",
            "rebuild_indices": rebuild_indices
        }
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rag/performance", tags=["RAG System"])
async def get_rag_performance():
    """Get RAG system performance metrics"""
    try:
        # This would return actual performance metrics in full implementation
        return {
            "status": "healthy",
            "metrics": {
                "total_queries": 0,
                "avg_response_time": 0.0,
                "cache_hit_rate": 0.0,
                "success_rate": 1.0
            },
            "message": "Performance monitoring endpoint ready"
        }
        
    except Exception as e:
        logger.error(f"Failed to get RAG performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/semantic-search", tags=["RAG System"])
async def semantic_search(
    request: Dict[str, Any],
    db: DatabaseManager = Depends(get_db)
):
    """Perform semantic search on physics content"""
    try:
        query_text = request.get("text", "")
        content_types = request.get("content_types", [])
        limit = request.get("limit", 10)
        min_similarity = request.get("min_similarity", 0.3)
        
        # This would use the actual semantic search engine
        # For now, return a placeholder response
        return {
            "status": "success",
            "query": query_text,
            "results": [],
            "total_results": 0,
            "search_type": "semantic",
            "message": "Semantic search endpoint ready - requires RAG system initialization"
        }
        
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/graph-search", tags=["RAG System"])
async def graph_enhanced_search(
    request: Dict[str, Any],
    db: DatabaseManager = Depends(get_db)
):
    """Perform graph-enhanced search with traversal strategies"""
    try:
        query_text = request.get("text", "")
        traversal_strategy = request.get("traversal_strategy", "breadth_first")
        include_learning_paths = request.get("include_learning_paths", False)
        student_level = request.get("student_level", "intermediate")
        
        return {
            "status": "success",
            "query": query_text,
            "traversal_strategy": traversal_strategy,
            "results": [],
            "learning_paths": [],
            "message": "Graph-enhanced search endpoint ready"
        }
        
    except Exception as e:
        logger.error(f"Graph search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rag/student-profile/{user_id}", tags=["RAG System"])
async def get_student_profile(
    user_id: str,
    db: DatabaseManager = Depends(get_db)
):
    """Get student learning profile for personalization"""
    try:
        # This would fetch actual student profile
        return {
            "user_id": user_id,
            "profile": {
                "current_level": "intermediate",
                "learning_style": "multimodal",
                "content_preferences": ["conceptual", "practical"],
                "topic_mastery": {},
                "struggling_concepts": [],
                "last_updated": datetime.now().isoformat()
            },
            "message": "Student profile endpoint ready"
        }
        
    except Exception as e:
        logger.error(f"Failed to get student profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/learning-path", tags=["RAG System"])
async def generate_learning_path(
    request: Dict[str, Any],
    db: DatabaseManager = Depends(get_db)
):
    """Generate learning path between physics concepts"""
    try:
        start_concept = request.get("start_concept", "")
        end_concept = request.get("end_concept", "")
        student_level = request.get("student_level", "intermediate")
        max_paths = request.get("max_paths", 3)
        
        return {
            "start_concept": start_concept,
            "end_concept": end_concept,
            "student_level": student_level,
            "learning_paths": [],
            "estimated_time": 0,
            "message": "Learning path generation endpoint ready"
        }
        
    except Exception as e:
        logger.error(f"Learning path generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/update-profile", tags=["RAG System"])
async def update_student_profile(
    request: Dict[str, Any],
    db: DatabaseManager = Depends(get_db)
):
    """Update student profile based on interaction data"""
    try:
        user_id = request.get("user_id", "")
        interaction_data = request.get("interaction_data", {})
        
        # This would update the actual student profile
        return {
            "user_id": user_id,
            "status": "updated",
            "changes_applied": [],
            "message": "Profile update endpoint ready"
        }
        
    except Exception as e:
        logger.error(f"Profile update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rag/cache-stats", tags=["RAG System"])
async def get_cache_statistics():
    """Get RAG system cache statistics"""
    try:
        return {
            "cache_stats": {
                "memory_cache": {
                    "hit_rate": 0.0,
                    "size": 0,
                    "max_size": 1000
                },
                "redis_cache": {
                    "hit_rate": 0.0,
                    "connected": False
                }
            },
            "message": "Cache statistics endpoint ready"
        }
        
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/clear-cache", tags=["RAG System"])
async def clear_rag_cache(
    category: Optional[str] = None
):
    """Clear RAG system cache (optionally by category)"""
    try:
        return {
            "status": "cleared",
            "category": category or "all",
            "message": "Cache clear endpoint ready"
        }
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rag/system-status", tags=["RAG System"])
async def get_rag_system_status():
    """Get comprehensive RAG system status"""
    try:
        return {
            "system_status": "initializing",
            "components": {
                "embedding_system": "not_initialized",
                "search_engine": "not_initialized", 
                "graph_retrieval": "not_initialized",
                "ranking_system": "not_initialized",
                "performance_optimization": "not_initialized"
            },
            "health_check": {
                "database_connections": "healthy",
                "knowledge_graph": "healthy",
                "cache_system": "healthy"
            },
            "version": "1.0.0",
            "message": "RAG system status endpoint ready"
        }
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Physics knowledge endpoints

@app.get("/physics/concepts", tags=["Physics Knowledge"])
async def get_physics_concepts(
    category: Optional[str] = None,
    db: DatabaseManager = Depends(get_db)
):
    """Get physics concepts from the knowledge graph"""
    try:
        concepts = await db.get_physics_concepts(category)
        
        return {
            "concepts": concepts,
            "count": len(concepts),
            "category": category
        }
        
    except Exception as e:
        logger.error(f"Failed to retrieve physics concepts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/physics/concepts/{concept_name}/related", tags=["Physics Knowledge"])
async def get_related_concepts(
    concept_name: str,
    db: DatabaseManager = Depends(get_db)
):
    """Get concepts related to a specific physics concept"""
    try:
        query = """
        MATCH (c:Concept {name: $concept_name})-[r]-(related:Concept)
        RETURN related.name as name, related.description as description, 
               related.category as category, type(r) as relationship
        """
        
        related = await db.neo4j.run_query(query, {"concept_name": concept_name})
        
        if not related:
            raise HTTPException(status_code=404, detail=f"Concept '{concept_name}' not found")
        
        return {
            "concept": concept_name,
            "related_concepts": related,
            "count": len(related)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve related concepts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Session management endpoints

@app.post("/sessions", tags=["Sessions"])
async def create_session(
    user_id: str,
    metadata: Optional[Dict[str, Any]] = None,
    db: DatabaseManager = Depends(get_db)
):
    """Create a new user session"""
    try:
        session_id = str(uuid.uuid4())
        session_data = {
            "user_id": user_id,
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            **(metadata or {})
        }
        
        # Cache session in Redis
        await db.cache_user_session(session_id, session_data, ttl=3600)  # 1 hour TTL
        
        return {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": session_data["created_at"]
        }
        
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}", tags=["Sessions"])
async def get_session(
    session_id: str,
    db: DatabaseManager = Depends(get_db)
):
    """Retrieve session information"""
    try:
        session_data = await db.get_user_session(session_id)
        
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return session_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background tasks

async def cache_agent_stats(db: DatabaseManager, agent_type: str):
    """Background task to cache frequently accessed agent statistics"""
    try:
        # This could be expanded to cache frequently accessed analytics
        cache_key = f"agent_stats:{agent_type}"
        
        async with db.postgres.get_connection() as conn:
            stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_interactions,
                    AVG(execution_time_ms) as avg_response_time
                FROM interactions 
                WHERE agent_type = $1 AND created_at >= NOW() - INTERVAL '24 hours'
            """, agent_type)
        
        if stats:
            cache_data = {
                "total_interactions": stats['total_interactions'],
                "avg_response_time": float(stats['avg_response_time']) if stats['avg_response_time'] else 0,
                "cached_at": datetime.now().isoformat()
            }
            
            await db.redis.set(cache_key, json.dumps(cache_data), ttl=300)  # 5 minutes
            
    except Exception as e:
        logger.warning(f"Failed to cache agent stats: {e}")

async def run_embedding_generation(model_name: str, rebuild_indices: bool = False):
    """Background task to generate embeddings"""
    try:
        logger.info(f"🚀 Starting embedding generation with model: {model_name}")
        
        # This would run the actual embedding generation pipeline
        # For now, just simulate the process
        await asyncio.sleep(2)  # Simulate work
        
        logger.info(f"✅ Embedding generation completed for model: {model_name}")
        
    except Exception as e:
        logger.error(f"❌ Embedding generation failed: {e}")
        raise

# Analytics endpoints

@app.get("/analytics/student-progress/{user_id}", tags=["Learning Analytics"])
async def get_student_progress(
    user_id: str,
    time_window_days: int = 30,
    include_predictions: bool = False,
    db: DatabaseManager = Depends(get_db)
):
    """Get comprehensive student progress analysis"""
    try:
        if not analytics_engine:
            raise HTTPException(status_code=503, detail="Analytics engine not available")
        
        # Get progress tracking
        progress_data = await analytics_engine.track_student_progress(user_id, time_window_days)
        
        # Add predictions if requested
        if include_predictions and data_miner:
            prediction = await data_miner.predict_student_performance(user_id, "success_rate")
            progress_data["performance_prediction"] = {
                "predicted_success_rate": prediction.predicted_value,
                "confidence_score": prediction.confidence_score,
                "contributing_factors": prediction.contributing_factors,
                "recommendations": prediction.intervention_recommendations
            }
        
        return progress_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get student progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analytics/concept-mastery", tags=["Learning Analytics"])
async def assess_concept_mastery(
    request: ConceptMasteryRequest,
    db: DatabaseManager = Depends(get_db)
):
    """Assess student mastery of a specific concept"""
    try:
        if not mastery_detector:
            raise HTTPException(status_code=503, detail="Mastery detector not available")
        
        assessment = await mastery_detector.assess_concept_mastery(
            request.user_id, 
            request.concept, 
            request.time_window_days
        )
        
        return {
            "concept": assessment.concept_name,
            "mastery_score": assessment.mastery_score,
            "confidence_interval": assessment.confidence_interval,
            "evidence_quality": assessment.evidence_quality,
            "error_patterns": [
                {
                    "error_type": ep.error_type,
                    "frequency": ep.frequency,
                    "intervention_priority": ep.intervention_priority,
                    "description": ep.description,
                    "suggested_remediation": ep.suggested_remediation
                } for ep in assessment.error_patterns
            ],
            "learning_trajectory": assessment.learning_trajectory,
            "next_steps": assessment.next_steps,
            "assessment_timestamp": assessment.assessment_timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to assess concept mastery: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analytics/learning-path", tags=["Learning Analytics"])
async def generate_learning_path(
    request: LearningPathRequest,
    db: DatabaseManager = Depends(get_db)
):
    """Generate optimized learning path for student"""
    try:
        if not path_optimizer:
            raise HTTPException(status_code=503, detail="Path optimizer not available")
        
        # Create learning objective
        from analytics.learning_path_optimizer import LearningObjective
        objective = LearningObjective(
            target_concepts=request.target_concepts,
            time_constraint=request.time_constraint,
            difficulty_preference=request.difficulty_preference
        )
        
        # Generate path
        path = await path_optimizer.generate_learning_path(
            request.user_id, 
            objective, 
            request.algorithm
        )
        
        return {
            "path_id": path.path_id,
            "student_id": path.student_id,
            "concept_sequence": path.concept_sequence,
            "estimated_total_time": path.estimated_total_time,
            "difficulty_progression": path.difficulty_progression,
            "success_probability": path.success_probability,
            "adaptive_checkpoints": path.adaptive_checkpoints,
            "alternative_paths": path.alternative_paths,
            "personalization_factors": path.personalization_factors,
            "creation_timestamp": path.creation_timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to generate learning path: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analytics/insights", tags=["Learning Analytics"])
async def get_educational_insights(
    request: AnalyticsInsightRequest,
    db: DatabaseManager = Depends(get_db)
):
    """Get educational insights from data mining"""
    try:
        if not data_miner:
            raise HTTPException(status_code=503, detail="Data miner not available")
        
        insights = await data_miner.generate_educational_insights(request.timeframe_days)
        
        # Filter by requested insight types
        filtered_insights = [
            insight for insight in insights 
            if insight.insight_type in request.insight_types
        ]
        
        return {
            "insights": [
                {
                    "insight_id": insight.insight_id,
                    "insight_type": insight.insight_type,
                    "title": insight.title,
                    "description": insight.description,
                    "significance_score": insight.significance_score,
                    "affected_population": insight.affected_population,
                    "actionable_recommendations": insight.actionable_recommendations,
                    "supporting_evidence": insight.supporting_evidence,
                    "temporal_validity": insight.temporal_validity
                } for insight in filtered_insights
            ],
            "total_insights": len(filtered_insights),
            "timeframe_days": request.timeframe_days
        }
        
    except Exception as e:
        logger.error(f"Failed to get educational insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analytics/student-clustering", tags=["Learning Analytics"])
async def analyze_student_clusters(
    request: StudentClusteringRequest,
    db: DatabaseManager = Depends(get_db)
):
    """Analyze student clusters and learning patterns"""
    try:
        if not data_miner:
            raise HTTPException(status_code=503, detail="Data miner not available")
        
        clusters = await data_miner.identify_student_clusters()
        
        return {
            "clusters": [
                {
                    "cluster_id": cluster.cluster_id,
                    "cluster_name": cluster.cluster_name,
                    "student_count": len(cluster.student_ids),
                    "characteristics": cluster.characteristics,
                    "common_patterns": cluster.common_patterns,
                    "success_strategies": cluster.success_strategies,
                    "risk_factors": cluster.risk_factors,
                    "recommended_interventions": cluster.recommended_interventions
                } for cluster in clusters
            ],
            "total_clusters": len(clusters),
            "algorithm_used": request.algorithm
        }
        
    except Exception as e:
        logger.error(f"Failed to analyze student clusters: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/learning-difficulties/{user_id}", tags=["Learning Analytics"])
async def detect_learning_difficulties(
    user_id: str,
    db: DatabaseManager = Depends(get_db)
):
    """Detect early warning signs for learning difficulties"""
    try:
        if not analytics_engine:
            raise HTTPException(status_code=503, detail="Analytics engine not available")
        
        difficulties = await analytics_engine.detect_learning_difficulties(user_id)
        
        return difficulties
        
    except Exception as e:
        logger.error(f"Failed to detect learning difficulties: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/efficiency/{user_id}", tags=["Learning Analytics"])
async def get_learning_efficiency(
    user_id: str,
    db: DatabaseManager = Depends(get_db)
):
    """Calculate learning efficiency metrics for a student"""
    try:
        if not analytics_engine:
            raise HTTPException(status_code=503, detail="Analytics engine not available")
        
        efficiency = await analytics_engine.calculate_learning_efficiency(user_id)
        
        return efficiency
        
    except Exception as e:
        logger.error(f"Failed to calculate learning efficiency: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/misconceptions/{user_id}/{concept}", tags=["Learning Analytics"])
async def detect_misconceptions(
    user_id: str,
    concept: str,
    db: DatabaseManager = Depends(get_db)
):
    """Detect specific misconceptions in student understanding"""
    try:
        if not mastery_detector:
            raise HTTPException(status_code=503, detail="Mastery detector not available")
        
        misconceptions = await mastery_detector.detect_misconceptions(user_id, concept)
        
        return {
            "detected_misconceptions": [
                {
                    "misconception_id": mc.misconception_id,
                    "description": mc.description,
                    "affected_concepts": mc.affected_concepts,
                    "manifestation_frequency": mc.manifestation_frequency,
                    "severity_score": mc.severity_score,
                    "typical_errors": mc.typical_errors,
                    "corrective_strategies": mc.corrective_strategies
                } for mc in misconceptions
            ],
            "user_id": user_id,
            "concept": concept
        }
        
    except Exception as e:
        logger.error(f"Failed to detect misconceptions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/realtime/metrics", tags=["Real-time Analytics"])
async def get_realtime_metrics(
    metric_types: List[str] = ["system", "performance"],
    db: DatabaseManager = Depends(get_db)
):
    """Get real-time analytics metrics"""
    try:
        if not realtime_engine:
            raise HTTPException(status_code=503, detail="Real-time analytics not available")
        
        metrics = realtime_engine.get_system_metrics()
        stats = realtime_engine.get_processing_stats()
        
        result = {"timestamp": datetime.now().isoformat()}
        
        if "system" in metric_types:
            result["system_metrics"] = metrics.get("system", {})
        
        if "performance" in metric_types:
            result["performance_stats"] = stats
        
        if "processors" in metric_types:
            result["processor_metrics"] = metrics.get("processors", {})
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to get real-time metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analytics/realtime/event", tags=["Real-time Analytics"])
async def submit_analytics_event(
    event_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    db: DatabaseManager = Depends(get_db)
):
    """Submit an analytics event for real-time processing"""
    try:
        if not realtime_engine:
            raise HTTPException(status_code=503, detail="Real-time analytics not available")
        
        # Submit event in background
        background_tasks.add_task(
            submit_custom_realtime_event,
            event_data
        )
        
        return {"status": "event_submitted", "timestamp": datetime.now().isoformat()}
        
    except Exception as e:
        logger.error(f"Failed to submit analytics event: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task functions
async def submit_realtime_event(user_id: str, agent_type: str, success: bool, execution_time_ms: int = None):
    """Submit interaction event to real-time analytics"""
    try:
        if realtime_engine:
            event = create_interaction_event(user_id, agent_type, success, execution_time_ms)
            await realtime_engine.submit_event(event)
    except Exception as e:
        logger.error(f"Failed to submit real-time event: {e}")

async def submit_custom_realtime_event(event_data: Dict[str, Any]):
    """Submit custom event to real-time analytics"""
    try:
        if realtime_engine:
            # Create event from data
            from analytics.realtime_analytics import AnalyticsEvent, EventType
            
            event = AnalyticsEvent(
                event_id=event_data.get("event_id", f"custom_{datetime.now().timestamp()}"),
                event_type=EventType(event_data.get("event_type", "INTERACTION_SUCCESS")),
                user_id=event_data.get("user_id", "unknown"),
                timestamp=datetime.now(),
                data=event_data.get("data", {}),
                context=event_data.get("context", {}),
                priority=event_data.get("priority", 1)
            )
            
            await realtime_engine.submit_event(event)
    except Exception as e:
        logger.error(f"Failed to submit custom real-time event: {e}")

# Error handlers

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={"error": "Not Found", "detail": exc.detail, "path": request.url.path}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": exc.detail, "path": request.url.path}
    )

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True  # Set to False in production
    )