#!/usr/bin/env python3
"""
Physics Assistant Dashboard API Server
Specialized backend APIs optimized for analytics dashboard consumption
with advanced caching, data aggregation, and real-time streaming capabilities.
"""

import asyncio
import json
import uuid
import hashlib
import time
import gzip
import pickle
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Set
from concurrent.futures import ThreadPoolExecutor
import logging

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import redis.asyncio as redis
from sse_starlette.sse import EventSourceResponse
import pandas as pd
import numpy as np

# Import existing analytics infrastructure
from db_manager import DatabaseManager, get_db_manager
from api_server import analytics_engine, mastery_detector, path_optimizer, data_miner, realtime_engine

# Import advanced analytics engines
from analytics.predictive_analytics import PredictiveAnalyticsEngine
from analytics.comparative_analytics import ComparativeAnalyticsEngine
from analytics.content_effectiveness import ContentEffectivenessEngine
from analytics.statistical_analysis import StatisticalAnalysisEngine
from analytics.automated_insights import AutomatedInsightsEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics for dashboard API
DASHBOARD_REQUEST_COUNT = Counter(
    'dashboard_requests_total',
    'Total dashboard API requests',
    ['endpoint', 'cache_status']
)
DASHBOARD_RESPONSE_TIME = Histogram(
    'dashboard_response_seconds',
    'Dashboard API response time',
    ['endpoint', 'data_type']
)
CACHE_HIT_RATE = Gauge(
    'dashboard_cache_hit_rate',
    'Dashboard cache hit rate',
    ['cache_layer']
)
ACTIVE_WEBSOCKETS = Gauge(
    'dashboard_websockets_active',
    'Active WebSocket connections'
)
DATA_AGGREGATION_TIME = Histogram(
    'dashboard_aggregation_seconds',
    'Time spent on data aggregation',
    ['aggregation_type']
)

# FastAPI app configuration
app = FastAPI(
    title="Physics Assistant Dashboard API",
    description="Advanced backend APIs for analytics dashboard with caching and streaming",
    version="2.0.0",
    docs_url="/dashboard/docs",
    redoc_url="/dashboard/redoc"
)

# Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Security
security = HTTPBearer(auto_error=False)

# Global cache and connection managers
redis_client: Optional[redis.Redis] = None
memory_cache: Dict[str, Any] = {}
cache_stats: Dict[str, Dict] = {
    "memory": {"hits": 0, "misses": 0, "size": 0},
    "redis": {"hits": 0, "misses": 0, "size": 0},
    "database": {"hits": 0, "misses": 0}
}

# Advanced analytics engines
predictive_engine: Optional[PredictiveAnalyticsEngine] = None
comparative_engine: Optional[ComparativeAnalyticsEngine] = None
content_engine: Optional[ContentEffectivenessEngine] = None
statistical_engine: Optional[StatisticalAnalysisEngine] = None
insights_engine: Optional[AutomatedInsightsEngine] = None

# WebSocket connection manager
class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_connections: Dict[str, List[WebSocket]] = defaultdict(list)
        
    async def connect(self, websocket: WebSocket, user_id: str = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        if user_id:
            self.user_connections[user_id].append(websocket)
        ACTIVE_WEBSOCKETS.set(len(self.active_connections))
        
    def disconnect(self, websocket: WebSocket, user_id: str = None):
        self.active_connections.remove(websocket)
        if user_id and websocket in self.user_connections[user_id]:
            self.user_connections[user_id].remove(websocket)
        ACTIVE_WEBSOCKETS.set(len(self.active_connections))
        
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
        
    async def send_to_user(self, message: str, user_id: str):
        for connection in self.user_connections[user_id]:
            try:
                await connection.send_text(message)
            except:
                self.disconnect(connection, user_id)
                
    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                disconnected.append(connection)
        
        for connection in disconnected:
            self.disconnect(connection)

websocket_manager = WebSocketManager()

# Real-time event queue for SSE
event_queue: deque = deque(maxlen=1000)
background_processor_running = False

# Cache Management Classes
class CacheKey:
    """Generate consistent cache keys"""
    
    @staticmethod
    def dashboard_summary(user_id: str = None, time_range: str = "7d") -> str:
        return f"dashboard:summary:{user_id or 'all'}:{time_range}"
    
    @staticmethod
    def student_progress(user_id: str, days: int = 30) -> str:
        return f"progress:{user_id}:{days}"
    
    @staticmethod
    def concept_mastery(user_id: str, concept: str) -> str:
        return f"mastery:{user_id}:{concept}"
    
    @staticmethod
    def time_series(metric: str, granularity: str, start: datetime, end: datetime) -> str:
        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")
        return f"timeseries:{metric}:{granularity}:{start_str}:{end_str}"
    
    @staticmethod
    def aggregation(agg_type: str, filters: Dict[str, Any]) -> str:
        filter_hash = hashlib.md5(json.dumps(filters, sort_keys=True).encode()).hexdigest()[:8]
        return f"agg:{agg_type}:{filter_hash}"

class MultiLayerCache:
    """Advanced multi-layer caching system"""
    
    def __init__(self):
        self.memory_cache = {}
        self.memory_ttl = {}
        self.memory_max_size = 1000
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache (memory -> Redis -> None)"""
        
        # Check memory cache first
        if key in self.memory_cache:
            if key in self.memory_ttl and datetime.now() > self.memory_ttl[key]:
                del self.memory_cache[key]
                del self.memory_ttl[key]
                cache_stats["memory"]["misses"] += 1
            else:
                cache_stats["memory"]["hits"] += 1
                return self.memory_cache[key]
        
        # Check Redis cache
        if redis_client:
            try:
                cached_data = await redis_client.get(key)
                if cached_data:
                    cache_stats["redis"]["hits"] += 1
                    # Promote to memory cache
                    data = pickle.loads(cached_data)
                    await self.set_memory(key, data, ttl=300)  # 5 minutes in memory
                    return data
                else:
                    cache_stats["redis"]["misses"] += 1
            except Exception as e:
                logger.error(f"Redis cache error: {e}")
        
        cache_stats["memory"]["misses"] += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Set value in both memory and Redis cache"""
        await self.set_memory(key, value, ttl=min(ttl, 900))  # Max 15 minutes in memory
        
        if redis_client:
            try:
                await redis_client.setex(key, ttl, pickle.dumps(value))
            except Exception as e:
                logger.error(f"Redis cache set error: {e}")
    
    async def set_memory(self, key: str, value: Any, ttl: int) -> None:
        """Set value in memory cache with size management"""
        if len(self.memory_cache) >= self.memory_max_size:
            # Remove oldest entries
            oldest_keys = list(self.memory_cache.keys())[:50]
            for old_key in oldest_keys:
                del self.memory_cache[old_key]
                if old_key in self.memory_ttl:
                    del self.memory_ttl[old_key]
        
        self.memory_cache[key] = value
        self.memory_ttl[key] = datetime.now() + timedelta(seconds=ttl)
        cache_stats["memory"]["size"] = len(self.memory_cache)
    
    async def invalidate(self, pattern: str) -> None:
        """Invalidate cache entries matching pattern"""
        # Memory cache
        keys_to_remove = [k for k in self.memory_cache.keys() if pattern in k]
        for key in keys_to_remove:
            del self.memory_cache[key]
            if key in self.memory_ttl:
                del self.memory_ttl[key]
        
        # Redis cache  
        if redis_client:
            try:
                keys = await redis_client.keys(f"*{pattern}*")
                if keys:
                    await redis_client.delete(*keys)
            except Exception as e:
                logger.error(f"Redis cache invalidation error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        memory_stats = cache_stats["memory"]
        redis_stats = cache_stats["redis"]
        
        memory_total = memory_stats["hits"] + memory_stats["misses"]
        redis_total = redis_stats["hits"] + redis_stats["misses"]
        
        return {
            "memory": {
                "hit_rate": memory_stats["hits"] / max(memory_total, 1),
                "size": memory_stats["size"],
                "max_size": self.memory_max_size
            },
            "redis": {
                "hit_rate": redis_stats["hits"] / max(redis_total, 1),
                "connected": redis_client is not None
            },
            "total_requests": memory_total + redis_total
        }

cache = MultiLayerCache()

# Pydantic Models for Dashboard API
class DashboardTimeRange(BaseModel):
    """Time range specification for dashboard queries"""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    preset: Optional[str] = Field(None, regex="^(1h|6h|24h|7d|30d|90d|1y)$")
    
    def get_dates(self) -> Tuple[datetime, datetime]:
        """Convert to actual datetime range"""
        if self.start_date and self.end_date:
            return self.start_date, self.end_date
        
        end = datetime.now()
        if self.preset == "1h":
            start = end - timedelta(hours=1)
        elif self.preset == "6h":
            start = end - timedelta(hours=6)
        elif self.preset == "24h":
            start = end - timedelta(days=1)
        elif self.preset == "7d":
            start = end - timedelta(days=7)
        elif self.preset == "30d":
            start = end - timedelta(days=30)
        elif self.preset == "90d":
            start = end - timedelta(days=90)
        elif self.preset == "1y":
            start = end - timedelta(days=365)
        else:
            start = end - timedelta(days=7)  # Default to 7 days
        
        return start, end

class DashboardFilters(BaseModel):
    """Advanced filtering options for dashboard data"""
    user_ids: Optional[List[str]] = None
    agent_types: Optional[List[str]] = None
    concepts: Optional[List[str]] = None
    difficulty_levels: Optional[List[str]] = None
    success_only: Optional[bool] = None
    min_interactions: Optional[int] = None

class TimeSeriesRequest(BaseModel):
    """Request for time-series data"""
    metrics: List[str] = Field(..., description="Metrics to retrieve")
    time_range: DashboardTimeRange
    granularity: str = Field("1h", regex="^(5m|15m|1h|6h|1d|1w)$")
    filters: Optional[DashboardFilters] = None
    aggregation: str = Field("avg", regex="^(avg|sum|min|max|count)$")

class AggregationRequest(BaseModel):
    """Request for data aggregation"""
    dimensions: List[str] = Field(..., description="Dimensions to group by")
    metrics: List[str] = Field(..., description="Metrics to aggregate")
    time_range: DashboardTimeRange
    filters: Optional[DashboardFilters] = None
    limit: int = Field(100, le=1000)

class ComparativeAnalysisRequest(BaseModel):
    """Request for comparative analysis"""
    comparison_type: str = Field(..., regex="^(students|concepts|time_periods|cohorts)$")
    primary_entities: List[str]
    comparison_entities: List[str]
    metrics: List[str]
    time_range: DashboardTimeRange

class RealTimeAlert(BaseModel):
    """Real-time alert configuration"""
    alert_id: str
    metric: str
    threshold: float
    comparison: str = Field(..., regex="^(gt|lt|gte|lte|eq)$")
    enabled: bool = True

# Advanced Analytics Request Models
class PredictiveAnalysisRequest(BaseModel):
    """Request for predictive analysis"""
    student_ids: Optional[List[str]] = None
    prediction_type: str = Field(..., regex="^(success|engagement|performance|risk)$")
    horizon_days: int = Field(7, ge=1, le=30)
    include_confidence_intervals: bool = True
    include_contributing_factors: bool = True

class ComparativeAnalysisRequest(BaseModel):
    """Request for comparative analysis"""
    analysis_type: str = Field(..., regex="^(cohort|temporal|ab_test|benchmark)$")
    primary_entities: List[str]
    comparison_entities: List[str]
    metrics: List[str]
    time_range: DashboardTimeRange
    statistical_tests: bool = True
    effect_size_calculation: bool = True

class ContentEffectivenessRequest(BaseModel):
    """Request for content effectiveness analysis"""
    content_ids: Optional[List[str]] = None
    content_types: Optional[List[str]] = None
    analysis_depth: str = Field("standard", regex="^(basic|standard|comprehensive)$")
    include_recommendations: bool = True
    time_window_days: int = Field(30, ge=7, le=90)

class StatisticalAnalysisRequest(BaseModel):
    """Request for statistical analysis"""
    analysis_type: str = Field(..., regex="^(timeseries|clustering|correlation|anomaly)$")
    metrics: List[str]
    time_range: DashboardTimeRange
    granularity: str = Field("1D", regex="^(1H|6H|1D|1W)$")
    advanced_options: Optional[Dict[str, Any]] = None

class InsightGenerationRequest(BaseModel):
    """Request for automated insight generation"""
    insight_types: List[str] = Field(default=["trend", "anomaly", "performance", "recommendation"])
    time_window_days: int = Field(7, ge=1, le=30)
    importance_threshold: str = Field("medium", regex="^(low|medium|high|critical)$")
    target_audience: str = Field("educator", regex="^(educator|administrator|student)$")
    include_natural_language: bool = True

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize dashboard API server"""
    global redis_client, background_processor_running
    global predictive_engine, comparative_engine, content_engine, statistical_engine, insights_engine
    
    logger.info("🚀 Starting Dashboard API Server")
    
    try:
        # Initialize Redis connection
        redis_client = redis.Redis(
            host="localhost",
            port=6379,
            decode_responses=False,
            socket_connect_timeout=5,
            socket_timeout=5
        )
        await redis_client.ping()
        logger.info("✅ Redis connection established")
        
    except Exception as e:
        logger.warning(f"⚠️ Redis connection failed: {e}")
        redis_client = None
    
    # Initialize advanced analytics engines
    try:
        db_manager = await get_db()
        
        predictive_engine = PredictiveAnalyticsEngine(db_manager)
        await predictive_engine.initialize()
        
        comparative_engine = ComparativeAnalyticsEngine(db_manager)
        await comparative_engine.initialize()
        
        content_engine = ContentEffectivenessEngine(db_manager)
        await content_engine.initialize()
        
        statistical_engine = StatisticalAnalysisEngine(db_manager)
        await statistical_engine.initialize()
        
        insights_engine = AutomatedInsightsEngine(db_manager)
        await insights_engine.initialize()
        
        logger.info("✅ Advanced analytics engines initialized")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize advanced analytics engines: {e}")
    
    # Start background processing
    if not background_processor_running:
        asyncio.create_task(background_event_processor())
        asyncio.create_task(cache_warming_task())
        background_processor_running = True
        logger.info("✅ Background processors started")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup dashboard API server"""
    global redis_client, background_processor_running
    
    logger.info("🔒 Shutting down Dashboard API Server")
    
    background_processor_running = False
    
    if redis_client:
        await redis_client.close()
        logger.info("✅ Redis connection closed")

# Dependency functions
async def get_db() -> DatabaseManager:
    """Get database manager"""
    if not hasattr(get_db, "db_manager") or not get_db.db_manager:
        get_db.db_manager = DatabaseManager()
        await get_db.db_manager.initialize()
    return get_db.db_manager

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token (placeholder for authentication)"""
    # In production, implement proper JWT token verification
    return True

# Core dashboard endpoints
@app.get("/dashboard/summary", tags=["Dashboard Core"])
async def get_dashboard_summary(
    time_range: DashboardTimeRange = Depends(),
    filters: DashboardFilters = Depends(),
    force_refresh: bool = False,
    db: DatabaseManager = Depends(get_db)
):
    """Get comprehensive dashboard summary with caching"""
    start_time = time.time()
    
    try:
        # Generate cache key
        cache_key = CacheKey.dashboard_summary(
            user_id=",".join(filters.user_ids) if filters.user_ids else None,
            time_range=time_range.preset or "custom"
        )
        
        # Check cache unless force refresh
        cached_data = None if force_refresh else await cache.get(cache_key)
        
        if cached_data:
            DASHBOARD_REQUEST_COUNT.labels(endpoint="summary", cache_status="hit").inc()
            CACHE_HIT_RATE.labels(cache_layer="multi").set(cache.get_stats()["memory"]["hit_rate"])
            return cached_data
        
        DASHBOARD_REQUEST_COUNT.labels(endpoint="summary", cache_status="miss").inc()
        
        # Generate summary data
        start_date, end_date = time_range.get_dates()
        
        async with db.postgres.get_connection() as conn:
            # Build base query
            base_query = """
            SELECT 
                COUNT(*) as total_interactions,
                COUNT(DISTINCT user_id) as active_users,
                COUNT(DISTINCT agent_type) as agents_used,
                AVG(execution_time_ms) as avg_response_time,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_interactions,
                AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate
            FROM interactions 
            WHERE created_at BETWEEN $1 AND $2
            """
            
            params = [start_date, end_date]
            
            # Add filters
            if filters.user_ids:
                base_query += f" AND user_id = ANY(${len(params) + 1})"
                params.append(filters.user_ids)
            
            if filters.agent_types:
                base_query += f" AND agent_type = ANY(${len(params) + 1})"
                params.append(filters.agent_types)
            
            # Execute main query
            summary_result = await conn.fetchrow(base_query, *params)
            
            # Get agent breakdown
            agent_query = base_query.replace("COUNT(*)", "agent_type, COUNT(*)") + " GROUP BY agent_type ORDER BY COUNT(*) DESC"
            agent_results = await conn.fetch(agent_query, *params)
            
            # Get hourly activity (last 24 hours)
            activity_query = """
            SELECT 
                date_trunc('hour', created_at) as hour,
                COUNT(*) as interactions,
                AVG(execution_time_ms) as avg_response_time
            FROM interactions 
            WHERE created_at >= NOW() - INTERVAL '24 hours'
            GROUP BY hour 
            ORDER BY hour
            """
            activity_results = await conn.fetch(activity_query)
        
        # Build response data
        response_data = {
            "time_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "preset": time_range.preset
            },
            "summary": {
                "total_interactions": summary_result['total_interactions'],
                "active_users": summary_result['active_users'],
                "agents_used": summary_result['agents_used'],
                "avg_response_time": float(summary_result['avg_response_time'] or 0),
                "success_rate": float(summary_result['success_rate'] or 0),
                "successful_interactions": summary_result['successful_interactions']
            },
            "agent_breakdown": [
                {
                    "agent_type": row['agent_type'],
                    "interaction_count": row['count']
                }
                for row in agent_results
            ],
            "hourly_activity": [
                {
                    "hour": row['hour'].isoformat(),
                    "interactions": row['interactions'],
                    "avg_response_time": float(row['avg_response_time'] or 0)
                }
                for row in activity_results
            ],
            "cache_info": {
                "generated_at": datetime.now().isoformat(),
                "cache_key": cache_key
            }
        }
        
        # Cache the result
        await cache.set(cache_key, response_data, ttl=300)  # 5 minutes
        
        DASHBOARD_RESPONSE_TIME.labels(endpoint="summary", data_type="aggregated").observe(time.time() - start_time)
        
        return response_data
        
    except Exception as e:
        logger.error(f"Dashboard summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dashboard/timeseries", tags=["Dashboard Core"])
async def get_time_series_data(
    request: TimeSeriesRequest,
    db: DatabaseManager = Depends(get_db)
):
    """Get time-series data for dashboard charts"""
    start_time = time.time()
    
    try:
        start_date, end_date = request.time_range.get_dates()
        cache_key = CacheKey.time_series(
            ",".join(request.metrics),
            request.granularity,
            start_date,
            end_date
        )
        
        # Check cache
        cached_data = await cache.get(cache_key)
        if cached_data:
            DASHBOARD_REQUEST_COUNT.labels(endpoint="timeseries", cache_status="hit").inc()
            return cached_data
        
        DASHBOARD_REQUEST_COUNT.labels(endpoint="timeseries", cache_status="miss").inc()
        
        # Generate time series data
        time_series_data = await generate_time_series(
            request.metrics,
            start_date,
            end_date,
            request.granularity,
            request.filters,
            request.aggregation,
            db
        )
        
        response_data = {
            "metrics": request.metrics,
            "granularity": request.granularity,
            "time_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "data": time_series_data,
            "generated_at": datetime.now().isoformat()
        }
        
        # Cache result
        await cache.set(cache_key, response_data, ttl=900)  # 15 minutes
        
        DATA_AGGREGATION_TIME.labels(aggregation_type="timeseries").observe(time.time() - start_time)
        
        return response_data
        
    except Exception as e:
        logger.error(f"Time series error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Continue with rest of the endpoints...
async def generate_time_series(
    metrics: List[str],
    start_date: datetime,
    end_date: datetime,
    granularity: str,
    filters: Optional[DashboardFilters],
    aggregation: str,
    db: DatabaseManager
) -> List[Dict[str, Any]]:
    """Generate time series data based on granularity"""
    
    # Map granularity to PostgreSQL interval
    interval_map = {
        "5m": "5 minutes",
        "15m": "15 minutes", 
        "1h": "1 hour",
        "6h": "6 hours",
        "1d": "1 day",
        "1w": "1 week"
    }
    
    interval = interval_map.get(granularity, "1 hour")
    
    async with db.postgres.get_connection() as conn:
        # Build time series query
        query = f"""
        WITH time_buckets AS (
            SELECT generate_series(
                date_trunc('{granularity.replace('m', 'min').replace('h', 'hour').replace('d', 'day').replace('w', 'week')}', $1),
                date_trunc('{granularity.replace('m', 'min').replace('h', 'hour').replace('d', 'day').replace('w', 'week')}', $2),
                interval '{interval}'
            ) as bucket
        )
        SELECT 
            tb.bucket,
            COALESCE(COUNT(i.id), 0) as interaction_count,
            COALESCE(AVG(i.execution_time_ms), 0) as avg_response_time,
            COALESCE(AVG(CASE WHEN i.success THEN 1.0 ELSE 0.0 END), 0) as success_rate,
            COALESCE(COUNT(DISTINCT i.user_id), 0) as unique_users
        FROM time_buckets tb
        LEFT JOIN interactions i ON date_trunc('{granularity.replace('m', 'min').replace('h', 'hour').replace('d', 'day').replace('w', 'week')}', i.created_at) = tb.bucket
            AND i.created_at BETWEEN $1 AND $2
        """
        
        params = [start_date, end_date]
        
        # Add filters
        if filters:
            if filters.user_ids:
                query += f" AND i.user_id = ANY(${len(params) + 1})"
                params.append(filters.user_ids)
            
            if filters.agent_types:
                query += f" AND i.agent_type = ANY(${len(params) + 1})"
                params.append(filters.agent_types)
        
        query += " GROUP BY tb.bucket ORDER BY tb.bucket"
        
        results = await conn.fetch(query, *params)
        
        return [
            {
                "timestamp": row['bucket'].isoformat(),
                "interaction_count": row['interaction_count'],
                "avg_response_time": float(row['avg_response_time']),
                "success_rate": float(row['success_rate']),
                "unique_users": row['unique_users']
            }
            for row in results
        ]

@app.post("/dashboard/aggregation", tags=["Dashboard Core"])
async def get_aggregated_data(
    request: AggregationRequest,
    db: DatabaseManager = Depends(get_db)
):
    """Get aggregated data for dashboard widgets"""
    start_time = time.time()
    
    try:
        start_date, end_date = request.time_range.get_dates()
        cache_key = CacheKey.aggregation(
            f"{','.join(request.dimensions)}_{','.join(request.metrics)}",
            {
                "time_range": request.time_range.dict(),
                "filters": request.filters.dict() if request.filters else {},
                "limit": request.limit
            }
        )
        
        # Check cache
        cached_data = await cache.get(cache_key)
        if cached_data:
            DASHBOARD_REQUEST_COUNT.labels(endpoint="aggregation", cache_status="hit").inc()
            return cached_data
        
        DASHBOARD_REQUEST_COUNT.labels(endpoint="aggregation", cache_status="miss").inc()
        
        # Generate aggregated data
        aggregated_data = await generate_aggregated_data(
            request.dimensions,
            request.metrics,
            start_date,
            end_date,
            request.filters,
            request.limit,
            db
        )
        
        response_data = {
            "dimensions": request.dimensions,
            "metrics": request.metrics,
            "time_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "data": aggregated_data,
            "total_records": len(aggregated_data),
            "generated_at": datetime.now().isoformat()
        }
        
        # Cache result
        await cache.set(cache_key, response_data, ttl=600)  # 10 minutes
        
        DATA_AGGREGATION_TIME.labels(aggregation_type="grouped").observe(time.time() - start_time)
        
        return response_data
        
    except Exception as e:
        logger.error(f"Aggregation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dashboard/comparative", tags=["Dashboard Analytics"])
async def get_comparative_analysis(
    request: ComparativeAnalysisRequest,
    db: DatabaseManager = Depends(get_db)
):
    """Get comparative analysis between entities"""
    start_time = time.time()
    
    try:
        start_date, end_date = request.time_range.get_dates()
        
        comparative_data = await generate_comparative_analysis(
            request.comparison_type,
            request.primary_entities,
            request.comparison_entities,
            request.metrics,
            start_date,
            end_date,
            db
        )
        
        response_data = {
            "comparison_type": request.comparison_type,
            "primary_entities": request.primary_entities,
            "comparison_entities": request.comparison_entities,
            "metrics": request.metrics,
            "time_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "comparisons": comparative_data,
            "generated_at": datetime.now().isoformat()
        }
        
        DATA_AGGREGATION_TIME.labels(aggregation_type="comparative").observe(time.time() - start_time)
        
        return response_data
        
    except Exception as e:
        logger.error(f"Comparative analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Real-time endpoints
@app.websocket("/dashboard/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time dashboard updates"""
    await websocket_manager.connect(websocket, user_id)
    
    try:
        while True:
            # Send periodic updates
            await asyncio.sleep(30)  # Update every 30 seconds
            
            # Get real-time metrics
            metrics = await get_realtime_dashboard_metrics(user_id)
            await websocket_manager.send_personal_message(
                json.dumps({
                    "type": "metrics_update",
                    "data": metrics,
                    "timestamp": datetime.now().isoformat()
                }),
                websocket
            )
            
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket, user_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        websocket_manager.disconnect(websocket, user_id)

@app.get("/dashboard/stream", tags=["Real-time"])
async def stream_dashboard_events(user_id: Optional[str] = None):
    """Server-Sent Events endpoint for real-time updates"""
    
    async def event_generator():
        last_sent = time.time()
        
        while True:
            # Check for new events every 5 seconds
            await asyncio.sleep(5)
            
            # Get events from queue
            current_time = time.time()
            events_to_send = []
            
            # Filter events for specific user if provided
            for event in list(event_queue):
                if user_id and event.get("user_id") != user_id:
                    continue
                if event.get("timestamp", 0) > last_sent:
                    events_to_send.append(event)
            
            if events_to_send:
                yield {
                    "event": "dashboard_update",
                    "data": json.dumps({
                        "events": events_to_send,
                        "timestamp": datetime.now().isoformat()
                    })
                }
                last_sent = current_time
            
            # Send heartbeat every 30 seconds
            if current_time - last_sent > 30:
                yield {
                    "event": "heartbeat",
                    "data": json.dumps({"timestamp": datetime.now().isoformat()})
                }
                last_sent = current_time
    
    return EventSourceResponse(event_generator())

# Advanced analytics endpoints
@app.get("/dashboard/student-insights/{user_id}", tags=["Dashboard Analytics"])
async def get_student_insights(
    user_id: str,
    time_range: DashboardTimeRange = Depends(),
    include_predictions: bool = False,
    db: DatabaseManager = Depends(get_db)
):
    """Get comprehensive student insights with caching"""
    try:
        cache_key = CacheKey.student_progress(user_id, 30)
        
        # Check cache
        cached_data = await cache.get(cache_key)
        if cached_data and not include_predictions:
            DASHBOARD_REQUEST_COUNT.labels(endpoint="student_insights", cache_status="hit").inc()
            return cached_data
        
        DASHBOARD_REQUEST_COUNT.labels(endpoint="student_insights", cache_status="miss").inc()
        
        start_date, end_date = time_range.get_dates()
        
        # Get student progress from analytics engine
        if analytics_engine:
            progress_data = await analytics_engine.track_student_progress(
                user_id, 
                (end_date - start_date).days
            )
        else:
            progress_data = {"error": "Analytics engine not available"}
        
        # Get concept mastery data
        concept_mastery = []
        if mastery_detector:
            # Get recent concepts
            async with db.postgres.get_connection() as conn:
                concepts = await conn.fetch("""
                    SELECT DISTINCT jsonb_array_elements_text(metadata->'concepts') as concept
                    FROM interactions 
                    WHERE user_id = $1 AND created_at >= $2
                    LIMIT 10
                """, user_id, start_date)
                
                for concept_row in concepts:
                    if concept_row['concept']:
                        mastery = await mastery_detector.assess_concept_mastery(
                            user_id, 
                            concept_row['concept'], 
                            7
                        )
                        concept_mastery.append({
                            "concept": mastery.concept_name,
                            "mastery_score": mastery.mastery_score,
                            "confidence": mastery.confidence_interval
                        })
        
        # Get learning path recommendations
        learning_paths = []
        if path_optimizer:
            # This would be more sophisticated in practice
            pass
        
        response_data = {
            "user_id": user_id,
            "time_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "progress_tracking": progress_data,
            "concept_mastery": concept_mastery,
            "learning_paths": learning_paths,
            "generated_at": datetime.now().isoformat()
        }
        
        # Add predictions if requested
        if include_predictions and data_miner:
            try:
                prediction = await data_miner.predict_student_performance(user_id, "success_rate")
                response_data["predictions"] = {
                    "success_rate": {
                        "predicted_value": prediction.predicted_value,
                        "confidence": prediction.confidence_score,
                        "factors": prediction.contributing_factors
                    }
                }
            except:
                response_data["predictions"] = {"error": "Prediction not available"}
        
        # Cache result (shorter TTL if predictions included)
        cache_ttl = 300 if include_predictions else 900
        await cache.set(cache_key, response_data, ttl=cache_ttl)
        
        return response_data
        
    except Exception as e:
        logger.error(f"Student insights error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dashboard/class-overview", tags=["Dashboard Analytics"])
async def get_class_overview(
    class_ids: Optional[List[str]] = None,
    time_range: DashboardTimeRange = Depends(),
    include_comparisons: bool = False,
    db: DatabaseManager = Depends(get_db)
):
    """Get class-level overview with comparative analytics"""
    try:
        start_date, end_date = time_range.get_dates()
        
        async with db.postgres.get_connection() as conn:
            # Get class statistics
            base_query = """
            SELECT 
                COUNT(DISTINCT user_id) as total_students,
                COUNT(*) as total_interactions,
                AVG(execution_time_ms) as avg_response_time,
                AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as class_success_rate,
                MODE() WITHIN GROUP (ORDER BY agent_type) as most_used_agent
            FROM interactions 
            WHERE created_at BETWEEN $1 AND $2
            """
            
            params = [start_date, end_date]
            
            if class_ids:
                # This would require a class mapping table in practice
                pass
            
            class_stats = await conn.fetchrow(base_query, *params)
            
            # Get student performance distribution
            performance_query = """
            SELECT 
                user_id,
                COUNT(*) as interaction_count,
                AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate,
                AVG(execution_time_ms) as avg_response_time
            FROM interactions 
            WHERE created_at BETWEEN $1 AND $2
            GROUP BY user_id
            ORDER BY success_rate DESC
            """
            
            student_performance = await conn.fetch(performance_query, *params)
            
            # Calculate percentiles
            success_rates = [float(row['success_rate']) for row in student_performance]
            if success_rates:
                percentiles = {
                    "25th": np.percentile(success_rates, 25),
                    "50th": np.percentile(success_rates, 50),
                    "75th": np.percentile(success_rates, 75),
                    "90th": np.percentile(success_rates, 90)
                }
            else:
                percentiles = {}
        
        response_data = {
            "time_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "class_statistics": {
                "total_students": class_stats['total_students'],
                "total_interactions": class_stats['total_interactions'],
                "avg_response_time": float(class_stats['avg_response_time'] or 0),
                "class_success_rate": float(class_stats['class_success_rate'] or 0),
                "most_used_agent": class_stats['most_used_agent']
            },
            "performance_distribution": {
                "percentiles": percentiles,
                "student_count": len(student_performance)
            },
            "top_performers": [
                {
                    "user_id": row['user_id'],
                    "success_rate": float(row['success_rate']),
                    "interaction_count": row['interaction_count']
                }
                for row in student_performance[:10]
            ],
            "generated_at": datetime.now().isoformat()
        }
        
        return response_data
        
    except Exception as e:
        logger.error(f"Class overview error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Export and data management endpoints
@app.post("/dashboard/export", tags=["Data Management"])
async def export_dashboard_data(
    export_format: str = Field(..., regex="^(csv|json|excel)$"),
    data_type: str = Field(..., regex="^(interactions|analytics|summary)$"),
    time_range: DashboardTimeRange = Depends(),
    filters: Optional[DashboardFilters] = None,
    db: DatabaseManager = Depends(get_db)
):
    """Export dashboard data in various formats"""
    try:
        start_date, end_date = time_range.get_dates()
        
        # Generate export data based on type
        if data_type == "interactions":
            export_data = await export_interactions_data(start_date, end_date, filters, db)
        elif data_type == "analytics":
            export_data = await export_analytics_data(start_date, end_date, filters, db)
        else:  # summary
            export_data = await export_summary_data(start_date, end_date, filters, db)
        
        # Format data based on requested format
        if export_format == "csv":
            # Convert to CSV
            import io
            output = io.StringIO()
            pd.DataFrame(export_data).to_csv(output, index=False)
            csv_content = output.getvalue()
            
            return StreamingResponse(
                io.StringIO(csv_content),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=dashboard_export_{data_type}.csv"}
            )
        
        elif export_format == "excel":
            # Convert to Excel
            import io
            output = io.BytesIO()
            pd.DataFrame(export_data).to_excel(output, index=False)
            output.seek(0)
            
            return StreamingResponse(
                output,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": f"attachment; filename=dashboard_export_{data_type}.xlsx"}
            )
        
        else:  # json
            return {
                "format": export_format,
                "data_type": data_type,
                "time_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "data": export_data,
                "export_timestamp": datetime.now().isoformat()
            }
        
    except Exception as e:
        logger.error(f"Export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Cache management endpoints
@app.get("/dashboard/cache/stats", tags=["Cache Management"])
async def get_cache_statistics():
    """Get cache performance statistics"""
    try:
        stats = cache.get_stats()
        
        # Add Redis-specific stats if available
        if redis_client:
            try:
                redis_info = await redis_client.info("memory")
                stats["redis"]["memory_usage"] = redis_info.get("used_memory_human", "N/A")
                stats["redis"]["max_memory"] = redis_info.get("maxmemory_human", "N/A")
            except:
                pass
        
        return {
            "cache_statistics": stats,
            "cache_layers": ["memory", "redis"],
            "total_cache_requests": stats["total_requests"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Cache stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dashboard/cache/invalidate", tags=["Cache Management"])
async def invalidate_cache(
    pattern: Optional[str] = None,
    cache_layer: str = Field("all", regex="^(memory|redis|all)$")
):
    """Invalidate cache entries"""
    try:
        if pattern:
            await cache.invalidate(pattern)
            message = f"Cache invalidated for pattern: {pattern}"
        else:
            # Clear all cache
            cache.memory_cache.clear()
            cache.memory_ttl.clear()
            if redis_client and cache_layer in ["redis", "all"]:
                await redis_client.flushdb()
            message = "All cache cleared"
        
        return {
            "status": "success",
            "message": message,
            "cache_layer": cache_layer,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Cache invalidation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dashboard/cache/warm", tags=["Cache Management"])
async def warm_cache(
    background_tasks: BackgroundTasks,
    cache_types: List[str] = Field(default=["summary", "timeseries"])
):
    """Warm cache with frequently accessed data"""
    try:
        background_tasks.add_task(perform_cache_warming, cache_types)
        
        return {
            "status": "started",
            "message": "Cache warming initiated",
            "cache_types": cache_types,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Cache warming error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Advanced Analytics Endpoints
@app.post("/dashboard/analytics/predictive", tags=["Advanced Analytics"])
async def perform_predictive_analysis(
    request: PredictiveAnalysisRequest,
    background_tasks: BackgroundTasks,
    db: DatabaseManager = Depends(get_db)
):
    """Perform predictive analytics analysis"""
    try:
        if not predictive_engine:
            raise HTTPException(status_code=503, detail="Predictive analytics engine not available")
        
        # Generate cache key
        cache_key = f"predictive_{request.prediction_type}_{hash(str(request.dict()))}"
        
        # Check cache
        cached_result = await cache.get(cache_key)
        if cached_result:
            DASHBOARD_REQUEST_COUNT.labels(endpoint="predictive", cache_status="hit").inc()
            return cached_result
        
        DASHBOARD_REQUEST_COUNT.labels(endpoint="predictive", cache_status="miss").inc()
        
        # Perform prediction analysis
        if request.student_ids:
            # Individual student predictions
            predictions = []
            for student_id in request.student_ids:
                if request.prediction_type == "success":
                    prediction = await predictive_engine.predict_student_success(student_id, request.horizon_days)
                elif request.prediction_type == "engagement":
                    prediction = await predictive_engine.predict_student_engagement(student_id, request.horizon_days)
                else:
                    raise HTTPException(status_code=400, detail=f"Unsupported prediction type: {request.prediction_type}")
                
                prediction_data = {
                    'student_id': prediction.student_id,
                    'prediction_type': prediction.prediction_type,
                    'predicted_value': prediction.predicted_value,
                    'confidence_score': prediction.confidence_score,
                    'risk_level': prediction.risk_level,
                    'recommendations': prediction.recommendations
                }
                
                if request.include_confidence_intervals:
                    prediction_data['confidence_interval'] = prediction.confidence_interval
                
                if request.include_contributing_factors:
                    prediction_data['contributing_factors'] = prediction.contributing_factors
                
                predictions.append(prediction_data)
            
            response_data = {
                'prediction_type': request.prediction_type,
                'horizon_days': request.horizon_days,
                'predictions': predictions,
                'generated_at': datetime.now().isoformat()
            }
        else:
            # Generate early warning alerts for all students
            alerts = await predictive_engine.generate_early_warning_alerts()
            
            response_data = {
                'prediction_type': 'early_warning',
                'alerts_generated': len(alerts),
                'alerts': [
                    {
                        'student_id': alert.student_id,
                        'alert_type': alert.alert_type,
                        'severity': alert.severity,
                        'predicted_outcome': alert.predicted_outcome,
                        'confidence': alert.confidence,
                        'recommended_actions': alert.recommended_actions
                    }
                    for alert in alerts
                ],
                'generated_at': datetime.now().isoformat()
            }
        
        # Cache result
        await cache.set(cache_key, response_data, ttl=1800)  # 30 minutes
        
        return response_data
        
    except Exception as e:
        logger.error(f"Predictive analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dashboard/analytics/comparative", tags=["Advanced Analytics"])
async def perform_comparative_analysis(
    request: ComparativeAnalysisRequest,
    db: DatabaseManager = Depends(get_db)
):
    """Perform comparative analytics analysis"""
    try:
        if not comparative_engine:
            raise HTTPException(status_code=503, detail="Comparative analytics engine not available")
        
        # Generate cache key
        cache_key = f"comparative_{request.analysis_type}_{hash(str(request.dict()))}"
        
        # Check cache
        cached_result = await cache.get(cache_key)
        if cached_result:
            DASHBOARD_REQUEST_COUNT.labels(endpoint="comparative", cache_status="hit").inc()
            return cached_result
        
        DASHBOARD_REQUEST_COUNT.labels(endpoint="comparative", cache_status="miss").inc()
        
        if request.analysis_type == "cohort":
            # Cohort comparison
            comparison_result = await comparative_engine.compare_cohorts(
                request.primary_entities[0],
                request.comparison_entities,
                request.metrics
            )
            
            response_data = {
                'analysis_type': 'cohort_comparison',
                'primary_cohort': comparison_result.primary_cohort,
                'comparison_cohorts': comparison_result.comparison_cohorts,
                'metrics_compared': comparison_result.metrics_compared,
                'statistical_results': comparison_result.statistical_results,
                'effect_sizes': comparison_result.effect_sizes,
                'practical_significance': comparison_result.practical_significance,
                'recommendations': comparison_result.recommendations,
                'confidence_level': comparison_result.confidence_level,
                'analysis_date': comparison_result.analysis_date.isoformat()
            }
            
        elif request.analysis_type == "benchmark":
            # Benchmark analysis against historical data
            start_date, end_date = request.time_range.get_dates()
            benchmarks = []
            
            for metric in request.metrics:
                benchmark = await comparative_engine.benchmark_against_historical(
                    metric, (end_date - start_date).days
                )
                
                benchmarks.append({
                    'metric_name': benchmark.metric_name,
                    'current_value': benchmark.current_value,
                    'historical_baseline': benchmark.historical_baseline,
                    'percentile_rank': benchmark.percentile_rank,
                    'trend_direction': benchmark.trend_direction,
                    'significance_level': benchmark.significance_level,
                    'is_outlier': benchmark.is_outlier
                })
            
            response_data = {
                'analysis_type': 'benchmark_analysis',
                'time_range': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                'benchmarks': benchmarks,
                'generated_at': datetime.now().isoformat()
            }
            
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported analysis type: {request.analysis_type}")
        
        # Cache result
        await cache.set(cache_key, response_data, ttl=3600)  # 1 hour
        
        return response_data
        
    except Exception as e:
        logger.error(f"Comparative analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dashboard/analytics/content-effectiveness", tags=["Advanced Analytics"])
async def analyze_content_effectiveness(
    request: ContentEffectivenessRequest,
    background_tasks: BackgroundTasks,
    db: DatabaseManager = Depends(get_db)
):
    """Analyze content effectiveness"""
    try:
        if not content_engine:
            raise HTTPException(status_code=503, detail="Content effectiveness engine not available")
        
        # Generate cache key
        cache_key = f"content_effectiveness_{hash(str(request.dict()))}"
        
        # Check cache
        cached_result = await cache.get(cache_key)
        if cached_result:
            DASHBOARD_REQUEST_COUNT.labels(endpoint="content_effectiveness", cache_status="hit").inc()
            return cached_result
        
        DASHBOARD_REQUEST_COUNT.labels(endpoint="content_effectiveness", cache_status="miss").inc()
        
        content_analyses = []
        
        if request.content_ids:
            # Analyze specific content items
            for content_id in request.content_ids:
                content_type = 'concept'  # Default, could be inferred or specified
                analysis = await content_engine.analyze_content_effectiveness(content_id, content_type)
                
                content_data = {
                    'content_id': analysis.content_id,
                    'content_type': analysis.content_type,
                    'engagement_score': analysis.engagement_score,
                    'learning_effectiveness': analysis.learning_effectiveness,
                    'difficulty_rating': analysis.difficulty_rating,
                    'completion_rate': analysis.completion_rate,
                    'success_rate': analysis.success_rate,
                    'time_to_mastery': analysis.time_to_mastery,
                    'student_satisfaction': analysis.student_satisfaction,
                    'interaction_count': analysis.interaction_count,
                    'unique_students': analysis.unique_students,
                    'error_patterns': analysis.error_patterns
                }
                
                if request.include_recommendations:
                    recommendations = await content_engine.generate_content_recommendations(content_id)
                    content_data['recommendations'] = [
                        {
                            'recommendation_type': rec.recommendation_type,
                            'priority': rec.priority,
                            'rationale': rec.rationale,
                            'suggested_changes': rec.suggested_changes,
                            'expected_impact': rec.expected_impact,
                            'success_probability': rec.success_probability
                        }
                        for rec in recommendations
                    ]
                
                content_analyses.append(content_data)
        else:
            # Analyze all available content
            # This would require getting content from the database
            # For now, return a placeholder
            content_analyses.append({
                'message': 'Content analysis requires specific content IDs',
                'available_content_types': ['concept', 'problem', 'explanation', 'example']
            })
        
        response_data = {
            'analysis_depth': request.analysis_depth,
            'time_window_days': request.time_window_days,
            'content_analyses': content_analyses,
            'total_content_analyzed': len(content_analyses),
            'generated_at': datetime.now().isoformat()
        }
        
        # Cache result
        await cache.set(cache_key, response_data, ttl=2400)  # 40 minutes
        
        return response_data
        
    except Exception as e:
        logger.error(f"Content effectiveness analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dashboard/analytics/statistical", tags=["Advanced Analytics"])
async def perform_statistical_analysis(
    request: StatisticalAnalysisRequest,
    db: DatabaseManager = Depends(get_db)
):
    """Perform advanced statistical analysis"""
    try:
        if not statistical_engine:
            raise HTTPException(status_code=503, detail="Statistical analysis engine not available")
        
        # Generate cache key
        cache_key = f"statistical_{request.analysis_type}_{hash(str(request.dict()))}"
        
        # Check cache
        cached_result = await cache.get(cache_key)
        if cached_result:
            DASHBOARD_REQUEST_COUNT.labels(endpoint="statistical", cache_status="hit").inc()
            return cached_result
        
        DASHBOARD_REQUEST_COUNT.labels(endpoint="statistical", cache_status="miss").inc()
        
        start_date, end_date = request.time_range.get_dates()
        
        if request.analysis_type == "timeseries":
            # Time series analysis
            time_series_results = []
            
            for metric in request.metrics:
                ts_analysis = await statistical_engine.analyze_time_series(
                    metric, start_date, end_date, request.granularity
                )
                
                time_series_results.append({
                    'metric_name': ts_analysis.metric_name,
                    'series_id': ts_analysis.series_id,
                    'time_period': {
                        'start': ts_analysis.time_period[0].isoformat(),
                        'end': ts_analysis.time_period[1].isoformat()
                    },
                    'trend_analysis': ts_analysis.trend_analysis,
                    'seasonality_analysis': ts_analysis.seasonality_analysis,
                    'stationarity_tests': ts_analysis.stationarity_tests,
                    'forecast_results': ts_analysis.forecast_results,
                    'anomalies': ts_analysis.anomalies,
                    'change_points': ts_analysis.change_points,
                    'model_performance': ts_analysis.model_performance
                })
            
            response_data = {
                'analysis_type': 'timeseries',
                'granularity': request.granularity,
                'time_series_analyses': time_series_results,
                'generated_at': datetime.now().isoformat()
            }
            
        elif request.analysis_type == "clustering":
            # Clustering analysis
            feature_set = ['success_rate', 'avg_response_time', 'interaction_frequency', 'concepts_covered']
            clustering_method = request.advanced_options.get('method', 'kmeans') if request.advanced_options else 'kmeans'
            
            cluster_analysis = await statistical_engine.perform_cluster_analysis(feature_set, clustering_method)
            
            response_data = {
                'analysis_type': 'clustering',
                'clustering_method': cluster_analysis.clustering_method,
                'feature_set': cluster_analysis.feature_set,
                'n_clusters': cluster_analysis.n_clusters,
                'cluster_quality_metrics': cluster_analysis.cluster_quality_metrics,
                'optimal_clusters': cluster_analysis.optimal_clusters,
                'feature_importance': cluster_analysis.feature_importance,
                'cluster_interpretations': cluster_analysis.cluster_interpretations,
                'generated_at': datetime.now().isoformat()
            }
            
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported analysis type: {request.analysis_type}")
        
        # Cache result
        await cache.set(cache_key, response_data, ttl=3600)  # 1 hour
        
        return response_data
        
    except Exception as e:
        logger.error(f"Statistical analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dashboard/analytics/insights", tags=["Advanced Analytics"])
async def generate_automated_insights(
    request: InsightGenerationRequest,
    db: DatabaseManager = Depends(get_db)
):
    """Generate automated insights and natural language summaries"""
    try:
        if not insights_engine:
            raise HTTPException(status_code=503, detail="Insights generation engine not available")
        
        # Generate cache key
        cache_key = f"insights_{hash(str(request.dict()))}"
        
        # Check cache
        cached_result = await cache.get(cache_key)
        if cached_result:
            DASHBOARD_REQUEST_COUNT.labels(endpoint="insights", cache_status="hit").inc()
            return cached_result
        
        DASHBOARD_REQUEST_COUNT.labels(endpoint="insights", cache_status="miss").inc()
        
        # Collect analytics data for insight generation
        end_date = datetime.now()
        start_date = end_date - timedelta(days=request.time_window_days)
        
        # Get summary data to feed into insights engine
        async with db.postgres.get_connection() as conn:
            summary_stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_interactions,
                    COUNT(DISTINCT user_id) as active_users,
                    AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate,
                    AVG(execution_time_ms) as avg_response_time
                FROM interactions 
                WHERE created_at BETWEEN $1 AND $2
            """, start_date, end_date)
        
        analytics_data = {
            'success_rate': [
                {'timestamp': start_date.isoformat(), 'value': float(summary_stats['success_rate'] or 0)},
                {'timestamp': end_date.isoformat(), 'value': float(summary_stats['success_rate'] or 0)}
            ],
            'total_interactions': int(summary_stats['total_interactions'] or 0),
            'active_users': int(summary_stats['active_users'] or 0),
            'avg_response_time': float(summary_stats['avg_response_time'] or 0)
        }
        
        # Generate insights
        insights = await insights_engine.generate_insights(analytics_data, request.time_window_days)
        
        # Filter by importance threshold
        importance_order = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        threshold_level = importance_order[request.importance_threshold]
        
        filtered_insights = [
            insight for insight in insights
            if importance_order.get(insight.importance_level, 0) >= threshold_level
        ]
        
        insights_data = [
            {
                'insight_id': insight.insight_id,
                'insight_type': insight.insight_type,
                'title': insight.title,
                'summary': insight.summary,
                'detailed_explanation': insight.detailed_explanation,
                'confidence_score': insight.confidence_score,
                'importance_level': insight.importance_level,
                'actionable_recommendations': insight.actionable_recommendations,
                'affected_entities': insight.affected_entities,
                'supporting_data': insight.supporting_data
            }
            for insight in filtered_insights
        ]
        
        response_data = {
            'insights_generated': len(insights_data),
            'insights': insights_data,
            'time_window_days': request.time_window_days,
            'importance_threshold': request.importance_threshold,
            'generated_at': datetime.now().isoformat()
        }
        
        # Generate natural language summary if requested
        if request.include_natural_language:
            nl_summary = await insights_engine.generate_natural_language_summary(
                analytics_data, 'class_overview', request.target_audience
            )
            
            response_data['natural_language_summary'] = {
                'headline': nl_summary.headline,
                'key_points': nl_summary.key_points,
                'narrative_text': nl_summary.narrative_text,
                'reading_level': nl_summary.reading_level,
                'length_category': nl_summary.length_category
            }
        
        # Cache result
        await cache.set(cache_key, response_data, ttl=1800)  # 30 minutes
        
        return response_data
        
    except Exception as e:
        logger.error(f"Insights generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Model Training and Background Processing Endpoints
@app.post("/dashboard/analytics/train-models", tags=["Advanced Analytics"])
async def trigger_model_training(
    background_tasks: BackgroundTasks,
    model_types: List[str] = Field(default=["predictive", "content_effectiveness"]),
    force_retrain: bool = False
):
    """Trigger background model training"""
    try:
        if not predictive_engine:
            raise HTTPException(status_code=503, detail="Analytics engines not available")
        
        # Add training tasks to background queue
        if "predictive" in model_types:
            background_tasks.add_task(train_predictive_models, force_retrain)
        
        if "content_effectiveness" in model_types:
            background_tasks.add_task(train_content_models, force_retrain)
        
        return {
            'status': 'training_initiated',
            'model_types': model_types,
            'force_retrain': force_retrain,
            'message': 'Model training tasks have been queued for background processing',
            'initiated_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Model training trigger error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background training functions
async def train_predictive_models(force_retrain: bool = False):
    """Background task to train predictive models"""
    try:
        if predictive_engine:
            logger.info("🤖 Starting predictive model training")
            
            # Train success prediction model
            success_performance = await predictive_engine.train_success_prediction_model()
            logger.info(f"✅ Success prediction model trained - Accuracy: {success_performance.accuracy:.3f}")
            
            # Train engagement prediction model
            engagement_performance = await predictive_engine.train_engagement_prediction_model()
            logger.info(f"✅ Engagement prediction model trained - R²: {engagement_performance.r2_score:.3f}")
            
    except Exception as e:
        logger.error(f"❌ Predictive model training failed: {e}")

async def train_content_models(force_retrain: bool = False):
    """Background task to train content effectiveness models"""
    try:
        if content_engine:
            logger.info("📚 Starting content effectiveness analysis")
            
            # Analyze key physics concepts
            concepts = ['kinematics', 'forces', 'energy', 'momentum', 'angular_motion']
            
            for concept in concepts:
                try:
                    await content_engine.analyze_content_effectiveness(concept, 'concept')
                    logger.info(f"✅ Content analysis completed for {concept}")
                except Exception as concept_error:
                    logger.error(f"❌ Content analysis failed for {concept}: {concept_error}")
            
    except Exception as e:
        logger.error(f"❌ Content model training failed: {e}")

# Mock data endpoints for development
@app.get("/dashboard/mock/student-progress", tags=["Mock Data"])
async def get_mock_student_progress():
    """Generate mock student progress data for frontend development"""
    return {
        "user_id": "mock_student_123",
        "progress_metrics": {
            "overall_score": 0.75,
            "concepts_mastered": 12,
            "total_concepts": 20,
            "learning_velocity": 0.65,
            "engagement_score": 0.82
        },
        "concept_breakdown": [
            {"concept": "Kinematics", "mastery": 0.89, "confidence": 0.85},
            {"concept": "Forces", "mastery": 0.72, "confidence": 0.78},
            {"concept": "Energy", "mastery": 0.65, "confidence": 0.70},
            {"concept": "Momentum", "mastery": 0.58, "confidence": 0.65}
        ],
        "time_series": [
            {"date": "2025-01-01", "score": 0.45},
            {"date": "2025-01-07", "score": 0.52},
            {"date": "2025-01-14", "score": 0.68},
            {"date": "2025-01-21", "score": 0.75}
        ],
        "mock_data": True
    }

@app.get("/dashboard/mock/class-analytics", tags=["Mock Data"])
async def get_mock_class_analytics():
    """Generate mock class analytics data for frontend development"""
    return {
        "class_id": "physics_101",
        "summary": {
            "total_students": 45,
            "active_students": 38,
            "avg_progress": 0.67,
            "completion_rate": 0.73
        },
        "performance_distribution": {
            "excellent": 8,
            "good": 15,
            "average": 12,
            "needs_help": 8,
            "at_risk": 2
        },
        "trending_concepts": [
            {"concept": "Newton's Laws", "difficulty": 0.45, "engagement": 0.78},
            {"concept": "Energy Conservation", "difficulty": 0.62, "engagement": 0.65},
            {"concept": "Circular Motion", "difficulty": 0.71, "engagement": 0.58}
        ],
        "mock_data": True
    }

# Background tasks and helper functions
async def generate_aggregated_data(
    dimensions: List[str],
    metrics: List[str],
    start_date: datetime,
    end_date: datetime,
    filters: Optional[DashboardFilters],
    limit: int,
    db: DatabaseManager
) -> List[Dict[str, Any]]:
    """Generate aggregated data based on dimensions and metrics"""
    
    # Build dynamic aggregation query
    select_parts = []
    group_parts = []
    
    # Add dimensions
    for dim in dimensions:
        if dim == "agent_type":
            select_parts.append("agent_type")
            group_parts.append("agent_type")
        elif dim == "user_id":
            select_parts.append("user_id")
            group_parts.append("user_id")
        elif dim == "date":
            select_parts.append("DATE(created_at) as date")
            group_parts.append("DATE(created_at)")
    
    # Add metrics
    for metric in metrics:
        if metric == "interaction_count":
            select_parts.append("COUNT(*) as interaction_count")
        elif metric == "avg_response_time":
            select_parts.append("AVG(execution_time_ms) as avg_response_time")
        elif metric == "success_rate":
            select_parts.append("AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate")
        elif metric == "unique_users":
            select_parts.append("COUNT(DISTINCT user_id) as unique_users")
    
    query = f"""
    SELECT {', '.join(select_parts)}
    FROM interactions 
    WHERE created_at BETWEEN $1 AND $2
    """
    
    params = [start_date, end_date]
    
    # Add filters
    if filters:
        if filters.user_ids:
            query += f" AND user_id = ANY(${len(params) + 1})"
            params.append(filters.user_ids)
        
        if filters.agent_types:
            query += f" AND agent_type = ANY(${len(params) + 1})"
            params.append(filters.agent_types)
        
        if filters.success_only:
            query += " AND success = true"
    
    if group_parts:
        query += f" GROUP BY {', '.join(group_parts)}"
    
    query += f" ORDER BY {select_parts[0]} LIMIT {limit}"
    
    async with db.postgres.get_connection() as conn:
        results = await conn.fetch(query, *params)
    
    return [dict(row) for row in results]

async def generate_comparative_analysis(
    comparison_type: str,
    primary_entities: List[str],
    comparison_entities: List[str],
    metrics: List[str],
    start_date: datetime,
    end_date: datetime,
    db: DatabaseManager
) -> List[Dict[str, Any]]:
    """Generate comparative analysis between entities"""
    
    comparisons = []
    
    if comparison_type == "students":
        # Compare student performance
        all_entities = primary_entities + comparison_entities
        
        async with db.postgres.get_connection() as conn:
            query = """
            SELECT 
                user_id,
                COUNT(*) as interaction_count,
                AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate,
                AVG(execution_time_ms) as avg_response_time
            FROM interactions 
            WHERE user_id = ANY($1) AND created_at BETWEEN $2 AND $3
            GROUP BY user_id
            """
            
            results = await conn.fetch(query, all_entities, start_date, end_date)
            
            for row in results:
                entity_type = "primary" if row['user_id'] in primary_entities else "comparison"
                comparisons.append({
                    "entity_id": row['user_id'],
                    "entity_type": entity_type,
                    "metrics": {
                        "interaction_count": row['interaction_count'],
                        "success_rate": float(row['success_rate']),
                        "avg_response_time": float(row['avg_response_time'] or 0)
                    }
                })
    
    elif comparison_type == "concepts":
        # Compare concept performance (would need concept metadata)
        pass
    
    return comparisons

async def get_realtime_dashboard_metrics(user_id: str) -> Dict[str, Any]:
    """Get real-time metrics for dashboard"""
    
    if realtime_engine:
        metrics = realtime_engine.get_system_metrics()
        stats = realtime_engine.get_processing_stats()
        
        return {
            "user_id": user_id,
            "system_metrics": metrics.get("system", {}),
            "processing_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
    
    return {
        "user_id": user_id,
        "message": "Real-time analytics not available",
        "timestamp": datetime.now().isoformat()
    }

async def export_interactions_data(
    start_date: datetime,
    end_date: datetime,
    filters: Optional[DashboardFilters],
    db: DatabaseManager
) -> List[Dict[str, Any]]:
    """Export interactions data"""
    
    async with db.postgres.get_connection() as conn:
        query = """
        SELECT 
            id, user_id, agent_type, created_at,
            success, execution_time_ms, metadata
        FROM interactions 
        WHERE created_at BETWEEN $1 AND $2
        """
        
        params = [start_date, end_date]
        
        if filters and filters.user_ids:
            query += f" AND user_id = ANY(${len(params) + 1})"
            params.append(filters.user_ids)
        
        query += " ORDER BY created_at DESC LIMIT 10000"  # Limit for performance
        
        results = await conn.fetch(query, *params)
    
    return [
        {
            "id": str(row['id']),
            "user_id": str(row['user_id']),
            "agent_type": row['agent_type'],
            "created_at": row['created_at'].isoformat(),
            "success": row['success'],
            "execution_time_ms": row['execution_time_ms'],
            "metadata": row['metadata']
        }
        for row in results
    ]

async def export_analytics_data(
    start_date: datetime,
    end_date: datetime,
    filters: Optional[DashboardFilters],
    db: DatabaseManager
) -> List[Dict[str, Any]]:
    """Export analytics data"""
    # This would export processed analytics data
    return [{"message": "Analytics export not yet implemented"}]

async def export_summary_data(
    start_date: datetime,
    end_date: datetime,
    filters: Optional[DashboardFilters],
    db: DatabaseManager
) -> List[Dict[str, Any]]:
    """Export summary data"""
    # This would export dashboard summary data
    return [{"message": "Summary export not yet implemented"}]

async def background_event_processor():
    """Background task to process real-time events"""
    global background_processor_running
    
    while background_processor_running:
        try:
            # Process events from real-time analytics engine
            if realtime_engine:
                # Get recent events and add to queue
                # This would integrate with the actual real-time engine
                pass
            
            await asyncio.sleep(5)  # Process every 5 seconds
            
        except Exception as e:
            logger.error(f"Background event processor error: {e}")
            await asyncio.sleep(10)

async def cache_warming_task():
    """Background task to warm frequently accessed cache entries"""
    global background_processor_running
    
    while background_processor_running:
        try:
            await asyncio.sleep(300)  # Warm cache every 5 minutes
            await perform_cache_warming(["summary", "timeseries"])
            
        except Exception as e:
            logger.error(f"Cache warming task error: {e}")
            await asyncio.sleep(600)

async def perform_cache_warming(cache_types: List[str]):
    """Perform cache warming for specified types"""
    try:
        logger.info(f"Starting cache warming for: {cache_types}")
        
        if "summary" in cache_types:
            # Warm dashboard summary cache
            cache_key = CacheKey.dashboard_summary(None, "7d")
            if not await cache.get(cache_key):
                # Generate and cache summary data
                logger.info("Warming dashboard summary cache")
        
        if "timeseries" in cache_types:
            # Warm common time-series queries
            common_metrics = ["interaction_count", "success_rate"]
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            cache_key = CacheKey.time_series(",".join(common_metrics), "1h", start_date, end_date)
            if not await cache.get(cache_key):
                logger.info("Warming time-series cache")
        
        logger.info("Cache warming completed")
        
    except Exception as e:
        logger.error(f"Cache warming error: {e}")

# Monitoring and health endpoints
@app.get("/dashboard/health", tags=["Monitoring"])
async def dashboard_health_check():
    """Health check for dashboard API"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "redis": "unknown",
                "cache": "healthy",
                "websockets": f"{len(websocket_manager.active_connections)} active",
                "background_processor": "running" if background_processor_running else "stopped"
            },
            "cache_stats": cache.get_stats()
        }
        
        # Check Redis connection
        if redis_client:
            try:
                await redis_client.ping()
                health_status["services"]["redis"] = "healthy"
            except:
                health_status["services"]["redis"] = "unhealthy"
                health_status["status"] = "degraded"
        else:
            health_status["services"]["redis"] = "not_configured"
        
        status_code = 200 if health_status["status"] == "healthy" else 206
        return JSONResponse(status_code=status_code, content=health_status)
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/dashboard/metrics", response_class=PlainTextResponse, tags=["Monitoring"])
async def get_dashboard_metrics():
    """Prometheus metrics endpoint for dashboard API"""
    return generate_latest()

# Error handlers
@app.exception_handler(404)
async def dashboard_not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Dashboard endpoint not found",
            "detail": exc.detail,
            "path": request.url.path,
            "available_endpoints": [
                "/dashboard/summary",
                "/dashboard/timeseries", 
                "/dashboard/aggregation",
                "/dashboard/ws/{user_id}",
                "/dashboard/stream"
            ]
        }
    )

@app.exception_handler(500)
async def dashboard_internal_error_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Dashboard internal server error",
            "detail": exc.detail,
            "path": request.url.path,
            "timestamp": datetime.now().isoformat()
        }
    )

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "dashboard_api_server:app",
        host="0.0.0.0",
        port=8001,
        log_level="info",
        reload=True
    )