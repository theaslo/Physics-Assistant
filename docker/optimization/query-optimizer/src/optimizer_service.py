#!/usr/bin/env python3
"""
Database Query Optimizer and Performance Monitor Service
Optimizes queries, manages caching, and provides performance insights
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

import asyncpg
import redis.asyncio as redis
from neo4j import AsyncGraphDatabase
import structlog
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from cachetools import TTLCache
import xxhash

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
QUERY_DURATION = Histogram('query_duration_seconds', 'Time spent executing queries', ['database', 'query_type'])
QUERY_COUNT = Counter('queries_total', 'Total number of queries', ['database', 'query_type', 'status'])
CACHE_HIT_RATE = Gauge('cache_hit_rate', 'Cache hit rate percentage', ['cache_type'])
SLOW_QUERY_COUNT = Counter('slow_queries_total', 'Number of slow queries', ['database'])
OPTIMIZATION_APPLIED = Counter('optimizations_applied_total', 'Number of optimizations applied', ['optimization_type'])

class DatabaseConnections:
    """Manages database connections"""
    
    def __init__(self):
        self.postgres_pool = None
        self.redis_client = None
        self.neo4j_driver = None
        
    async def initialize(self):
        """Initialize all database connections"""
        logger.info("Initializing database connections")
        
        # PostgreSQL connection pool
        postgres_dsn = f"postgresql://{os.getenv('POSTGRES_USER', 'postgres')}:" \
                      f"{os.getenv('POSTGRES_PASSWORD')}@" \
                      f"{os.getenv('POSTGRES_HOST', 'postgres')}:" \
                      f"{os.getenv('POSTGRES_PORT', '5432')}/" \
                      f"{os.getenv('POSTGRES_DB', 'physics_assistant')}"
        
        self.postgres_pool = await asyncpg.create_pool(
            postgres_dsn,
            min_size=5,
            max_size=20,
            command_timeout=30,
            server_settings={
                'jit': 'off',  # Disable JIT for consistent performance
                'shared_preload_libraries': 'pg_stat_statements',
            }
        )
        
        # Redis connection
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'redis'),
            port=int(os.getenv('REDIS_PORT', '6379')),
            password=os.getenv('REDIS_PASSWORD'),
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30
        )
        
        # Neo4j connection
        neo4j_uri = f"bolt://{os.getenv('NEO4J_HOST', 'neo4j')}:{os.getenv('NEO4J_PORT', '7687')}"
        self.neo4j_driver = AsyncGraphDatabase.driver(
            neo4j_uri,
            auth=(os.getenv('NEO4J_USER', 'neo4j'), os.getenv('NEO4J_PASSWORD'))
        )
        
        logger.info("Database connections initialized")
        
    async def close(self):
        """Close all database connections"""
        if self.postgres_pool:
            await self.postgres_pool.close()
        if self.redis_client:
            await self.redis_client.close()
        if self.neo4j_driver:
            await self.neo4j_driver.close()

class QueryAnalyzer:
    """Analyzes and optimizes database queries"""
    
    def __init__(self, db_connections: DatabaseConnections):
        self.db = db_connections
        self.query_cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour TTL
        self.slow_queries = []
        
    async def analyze_postgres_performance(self) -> Dict[str, Any]:
        """Analyze PostgreSQL performance metrics"""
        logger.info("Analyzing PostgreSQL performance")
        
        async with self.db.postgres_pool.acquire() as conn:
            # Get slow queries from pg_stat_statements
            slow_queries = await conn.fetch("""
                SELECT query, calls, total_exec_time, mean_exec_time, 
                       rows, 100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
                FROM pg_stat_statements 
                WHERE mean_exec_time > 100 
                ORDER BY mean_exec_time DESC 
                LIMIT 20
            """)
            
            # Get table statistics
            table_stats = await conn.fetch("""
                SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del, 
                       n_live_tup, n_dead_tup, last_vacuum, last_autovacuum, last_analyze
                FROM pg_stat_user_tables 
                ORDER BY n_live_tup DESC
            """)
            
            # Get index usage
            index_stats = await conn.fetch("""
                SELECT schemaname, tablename, indexname, idx_tup_read, idx_tup_fetch,
                       idx_tup_read - idx_tup_fetch as idx_tup_diff
                FROM pg_stat_user_indexes 
                WHERE idx_tup_read > 0
                ORDER BY idx_tup_read DESC
            """)
            
            # Get database size information
            db_size = await conn.fetch("""
                SELECT pg_database.datname, pg_size_pretty(pg_database_size(pg_database.datname)) AS size
                FROM pg_database
                WHERE pg_database.datname = current_database()
            """)
            
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'slow_queries': [dict(row) for row in slow_queries],
            'table_stats': [dict(row) for row in table_stats],
            'index_stats': [dict(row) for row in index_stats],
            'database_size': [dict(row) for row in db_size]
        }
    
    async def optimize_postgres_queries(self) -> List[Dict[str, Any]]:
        """Generate PostgreSQL query optimization recommendations"""
        logger.info("Generating PostgreSQL optimization recommendations")
        
        optimizations = []
        
        async with self.db.postgres_pool.acquire() as conn:
            # Check for missing indexes
            missing_indexes = await conn.fetch("""
                SELECT schemaname, tablename, attname, n_distinct, correlation
                FROM pg_stats 
                WHERE schemaname = 'public' 
                AND n_distinct > 100 
                AND correlation < 0.1
            """)
            
            for row in missing_indexes:
                optimizations.append({
                    'type': 'missing_index',
                    'severity': 'medium',
                    'table': f"{row['schemaname']}.{row['tablename']}",
                    'column': row['attname'],
                    'recommendation': f"CREATE INDEX idx_{row['tablename']}_{row['attname']} ON {row['tablename']} ({row['attname']});",
                    'reason': f"Column has {row['n_distinct']} distinct values with low correlation ({row['correlation']:.3f})"
                })
            
            # Check for unused indexes
            unused_indexes = await conn.fetch("""
                SELECT schemaname, tablename, indexname, idx_scan
                FROM pg_stat_user_indexes 
                WHERE idx_scan < 10
                AND indexname NOT LIKE '%_pkey'
            """)
            
            for row in unused_indexes:
                optimizations.append({
                    'type': 'unused_index',
                    'severity': 'low',
                    'table': f"{row['schemaname']}.{row['tablename']}",
                    'index': row['indexname'],
                    'recommendation': f"DROP INDEX IF EXISTS {row['indexname']};",
                    'reason': f"Index has only been scanned {row['idx_scan']} times"
                })
            
            # Check for tables needing VACUUM
            vacuum_needed = await conn.fetch("""
                SELECT schemaname, tablename, n_dead_tup, n_live_tup,
                       (n_dead_tup::float / GREATEST(n_live_tup::float, 1)) * 100 as dead_percent
                FROM pg_stat_user_tables 
                WHERE n_dead_tup > 1000 
                AND (n_dead_tup::float / GREATEST(n_live_tup::float, 1)) > 10
            """)
            
            for row in vacuum_needed:
                optimizations.append({
                    'type': 'vacuum_needed',
                    'severity': 'high' if row['dead_percent'] > 50 else 'medium',
                    'table': f"{row['schemaname']}.{row['tablename']}",
                    'recommendation': f"VACUUM ANALYZE {row['tablename']};",
                    'reason': f"Table has {row['dead_percent']:.1f}% dead tuples ({row['n_dead_tup']} dead, {row['n_live_tup']} live)"
                })
        
        OPTIMIZATION_APPLIED.labels(optimization_type='postgres').inc(len(optimizations))
        return optimizations
    
    async def analyze_redis_performance(self) -> Dict[str, Any]:
        """Analyze Redis performance metrics"""
        logger.info("Analyzing Redis performance")
        
        info = await self.redis_client.info()
        memory_info = await self.redis_client.info('memory')
        stats_info = await self.redis_client.info('stats')
        
        # Calculate hit rate
        hits = int(stats_info.get('keyspace_hits', 0))
        misses = int(stats_info.get('keyspace_misses', 0))
        total = hits + misses
        hit_rate = (hits / total * 100) if total > 0 else 0
        
        CACHE_HIT_RATE.labels(cache_type='redis').set(hit_rate)
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'memory_used': memory_info.get('used_memory_human'),
            'memory_peak': memory_info.get('used_memory_peak_human'),
            'hit_rate': hit_rate,
            'connected_clients': info.get('connected_clients'),
            'total_commands_processed': stats_info.get('total_commands_processed'),
            'instantaneous_ops_per_sec': stats_info.get('instantaneous_ops_per_sec'),
            'keyspace_hits': hits,
            'keyspace_misses': misses
        }
    
    async def optimize_redis_cache(self) -> List[Dict[str, Any]]:
        """Generate Redis optimization recommendations"""
        logger.info("Generating Redis optimization recommendations")
        
        optimizations = []
        info = await self.redis_client.info('memory')
        
        memory_usage = int(info.get('used_memory', 0))
        max_memory = int(info.get('maxmemory', 0))
        
        if max_memory > 0:
            memory_percent = (memory_usage / max_memory) * 100
            
            if memory_percent > 85:
                optimizations.append({
                    'type': 'memory_usage',
                    'severity': 'high',
                    'current_usage': f"{memory_percent:.1f}%",
                    'recommendation': "Consider increasing maxmemory or implementing more aggressive eviction policies",
                    'reason': f"Memory usage is at {memory_percent:.1f}% of maximum"
                })
            
            if memory_percent > 70:
                # Analyze key patterns for optimization
                cursor = 0
                large_keys = []
                
                while True:
                    cursor, keys = await self.redis_client.scan(cursor, count=100)
                    for key in keys:
                        memory_usage = await self.redis_client.memory_usage(key)
                        if memory_usage and memory_usage > 1024 * 1024:  # > 1MB
                            large_keys.append((key, memory_usage))
                    
                    if cursor == 0:
                        break
                
                if large_keys:
                    large_keys.sort(key=lambda x: x[1], reverse=True)
                    optimizations.append({
                        'type': 'large_keys',
                        'severity': 'medium',
                        'large_keys': large_keys[:10],  # Top 10 largest keys
                        'recommendation': "Review large keys for optimization opportunities",
                        'reason': f"Found {len(large_keys)} keys larger than 1MB"
                    })
        
        return optimizations

class CacheManager:
    """Manages intelligent caching strategies"""
    
    def __init__(self, db_connections: DatabaseConnections):
        self.db = db_connections
        self.cache_stats = {}
        
    def generate_cache_key(self, query: str, params: tuple = None) -> str:
        """Generate a consistent cache key for queries"""
        key_data = f"{query}:{params or ''}"
        return f"query_cache:{xxhash.xxh64(key_data.encode()).hexdigest()}"
    
    async def get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get result from cache"""
        try:
            cached_data = await self.db.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.warning("Cache retrieval failed", error=str(e))
        return None
    
    async def set_cached_result(self, cache_key: str, result: Any, ttl: int = 3600):
        """Set result in cache"""
        try:
            await self.db.redis_client.setex(
                cache_key, 
                ttl, 
                json.dumps(result, default=str)
            )
        except Exception as e:
            logger.warning("Cache storage failed", error=str(e))
    
    async def execute_cached_query(self, query: str, params: tuple = None, 
                                 ttl: int = 3600, force_refresh: bool = False) -> Any:
        """Execute query with caching"""
        cache_key = self.generate_cache_key(query, params)
        
        # Try cache first (unless forcing refresh)
        if not force_refresh:
            cached_result = await self.get_cached_result(cache_key)
            if cached_result is not None:
                CACHE_HIT_RATE.labels(cache_type='query').inc()
                return cached_result
        
        # Execute query
        start_time = time.time()
        try:
            async with self.db.postgres_pool.acquire() as conn:
                if params:
                    result = await conn.fetch(query, *params)
                else:
                    result = await conn.fetch(query)
                
                # Convert to serializable format
                result_data = [dict(row) for row in result]
                
                # Cache the result
                await self.set_cached_result(cache_key, result_data, ttl)
                
                duration = time.time() - start_time
                QUERY_DURATION.labels(database='postgres', query_type='cached').observe(duration)
                QUERY_COUNT.labels(database='postgres', query_type='cached', status='success').inc()
                
                return result_data
                
        except Exception as e:
            duration = time.time() - start_time
            QUERY_COUNT.labels(database='postgres', query_type='cached', status='error').inc()
            logger.error("Query execution failed", query=query, error=str(e), duration=duration)
            raise
    
    async def invalidate_cache_pattern(self, pattern: str):
        """Invalidate cache keys matching a pattern"""
        try:
            cursor = 0
            while True:
                cursor, keys = await self.db.redis_client.scan(cursor, match=pattern, count=100)
                if keys:
                    await self.db.redis_client.delete(*keys)
                
                if cursor == 0:
                    break
            
            logger.info("Cache invalidated", pattern=pattern)
        except Exception as e:
            logger.warning("Cache invalidation failed", pattern=pattern, error=str(e))

class PerformanceMonitor:
    """Monitors and analyzes performance metrics"""
    
    def __init__(self, db_connections: DatabaseConnections):
        self.db = db_connections
        self.metrics_history = []
        
    async def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics"""
        logger.info("Collecting system performance metrics")
        
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'postgres': {},
            'redis': {},
            'neo4j': {}
        }
        
        # PostgreSQL metrics
        try:
            async with self.db.postgres_pool.acquire() as conn:
                # Connection stats
                connection_stats = await conn.fetch("""
                    SELECT state, count(*) as count 
                    FROM pg_stat_activity 
                    WHERE datname = current_database() 
                    GROUP BY state
                """)
                
                # Lock stats
                lock_stats = await conn.fetch("""
                    SELECT mode, count(*) as count 
                    FROM pg_locks 
                    GROUP BY mode
                """)
                
                # Wait events
                wait_events = await conn.fetch("""
                    SELECT wait_event_type, wait_event, count(*) as count 
                    FROM pg_stat_activity 
                    WHERE wait_event IS NOT NULL 
                    GROUP BY wait_event_type, wait_event
                """)
                
                metrics['postgres'] = {
                    'connections': [dict(row) for row in connection_stats],
                    'locks': [dict(row) for row in lock_stats],
                    'wait_events': [dict(row) for row in wait_events]
                }
        except Exception as e:
            logger.error("Failed to collect PostgreSQL metrics", error=str(e))
        
        # Redis metrics
        try:
            redis_info = await self.db.redis_client.info()
            metrics['redis'] = {
                'connected_clients': redis_info.get('connected_clients'),
                'blocked_clients': redis_info.get('blocked_clients'),
                'used_memory': redis_info.get('used_memory'),
                'instantaneous_ops_per_sec': redis_info.get('instantaneous_ops_per_sec')
            }
        except Exception as e:
            logger.error("Failed to collect Redis metrics", error=str(e))
        
        # Neo4j metrics (if available)
        try:
            async with self.db.neo4j_driver.session() as session:
                result = await session.run("CALL dbms.queryJmx('*:*') YIELD attributes")
                neo4j_metrics = await result.data()
                metrics['neo4j'] = neo4j_metrics
        except Exception as e:
            logger.error("Failed to collect Neo4j metrics", error=str(e))
        
        self.metrics_history.append(metrics)
        
        # Keep only last 24 hours of metrics
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        self.metrics_history = [
            m for m in self.metrics_history 
            if datetime.fromisoformat(m['timestamp']) > cutoff_time
        ]
        
        return metrics
    
    async def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect performance anomalies using statistical analysis"""
        logger.info("Detecting performance anomalies")
        
        if len(self.metrics_history) < 10:
            return []  # Need sufficient data for analysis
        
        anomalies = []
        
        # Analyze PostgreSQL connection patterns
        try:
            connection_counts = []
            for metrics in self.metrics_history[-50:]:  # Last 50 measurements
                total_connections = sum(
                    conn['count'] for conn in metrics.get('postgres', {}).get('connections', [])
                )
                connection_counts.append(total_connections)
            
            if connection_counts:
                mean_connections = np.mean(connection_counts)
                std_connections = np.std(connection_counts)
                current_connections = connection_counts[-1]
                
                if abs(current_connections - mean_connections) > 2 * std_connections:
                    anomalies.append({
                        'type': 'connection_anomaly',
                        'severity': 'medium',
                        'current_value': current_connections,
                        'expected_range': f"{mean_connections - 2*std_connections:.1f} - {mean_connections + 2*std_connections:.1f}",
                        'description': f"Unusual number of database connections: {current_connections}"
                    })
        except Exception as e:
            logger.warning("Failed to analyze connection patterns", error=str(e))
        
        # Analyze Redis memory usage patterns
        try:
            memory_usage = []
            for metrics in self.metrics_history[-50:]:
                redis_memory = metrics.get('redis', {}).get('used_memory', 0)
                memory_usage.append(redis_memory)
            
            if memory_usage and any(m > 0 for m in memory_usage):
                # Check for sudden memory spikes
                memory_changes = np.diff(memory_usage)
                if len(memory_changes) > 0:
                    mean_change = np.mean(memory_changes)
                    std_change = np.std(memory_changes)
                    
                    if len(memory_changes) > 0 and abs(memory_changes[-1]) > mean_change + 3 * std_change:
                        anomalies.append({
                            'type': 'memory_spike',
                            'severity': 'high',
                            'memory_change': memory_changes[-1],
                            'description': f"Sudden Redis memory change: {memory_changes[-1]:,} bytes"
                        })
        except Exception as e:
            logger.warning("Failed to analyze memory patterns", error=str(e))
        
        return anomalies

class OptimizerService:
    """Main optimizer service orchestrator"""
    
    def __init__(self):
        self.db_connections = DatabaseConnections()
        self.query_analyzer = None
        self.cache_manager = None
        self.performance_monitor = None
        self.scheduler = AsyncIOScheduler()
        
    async def initialize(self):
        """Initialize the optimizer service"""
        logger.info("Initializing optimizer service")
        
        await self.db_connections.initialize()
        self.query_analyzer = QueryAnalyzer(self.db_connections)
        self.cache_manager = CacheManager(self.db_connections)
        self.performance_monitor = PerformanceMonitor(self.db_connections)
        
        # Schedule periodic tasks
        self.scheduler.add_job(
            self.periodic_analysis,
            'interval',
            minutes=15,
            id='periodic_analysis'
        )
        
        self.scheduler.add_job(
            self.cache_cleanup,
            'interval',
            hours=1,
            id='cache_cleanup'
        )
        
        self.scheduler.add_job(
            self.generate_optimization_report,
            'interval',
            hours=4,
            id='optimization_report'
        )
        
        self.scheduler.start()
        
        # Start Prometheus metrics server
        start_http_server(8080)
        
        logger.info("Optimizer service initialized")
    
    async def periodic_analysis(self):
        """Perform periodic performance analysis"""
        logger.info("Starting periodic analysis")
        
        try:
            # Collect metrics
            await self.performance_monitor.collect_system_metrics()
            
            # Generate optimizations
            postgres_opts = await self.query_analyzer.optimize_postgres_queries()
            redis_opts = await self.query_analyzer.optimize_redis_cache()
            
            # Detect anomalies
            anomalies = await self.performance_monitor.detect_anomalies()
            
            # Log findings
            if postgres_opts or redis_opts or anomalies:
                logger.info(
                    "Analysis completed",
                    postgres_optimizations=len(postgres_opts),
                    redis_optimizations=len(redis_opts),
                    anomalies=len(anomalies)
                )
            
        except Exception as e:
            logger.error("Periodic analysis failed", error=str(e))
    
    async def cache_cleanup(self):
        """Cleanup expired cache entries"""
        logger.info("Starting cache cleanup")
        
        try:
            # Get cache statistics
            redis_info = await self.db_connections.redis_client.info('memory')
            used_memory = int(redis_info.get('used_memory', 0))
            max_memory = int(redis_info.get('maxmemory', 0))
            
            if max_memory > 0:
                memory_percent = (used_memory / max_memory) * 100
                
                if memory_percent > 80:
                    # Aggressive cleanup for high memory usage
                    await self.cache_manager.invalidate_cache_pattern("query_cache:*")
                    logger.info("Aggressive cache cleanup performed", memory_percent=memory_percent)
                
        except Exception as e:
            logger.error("Cache cleanup failed", error=str(e))
    
    async def generate_optimization_report(self):
        """Generate comprehensive optimization report"""
        logger.info("Generating optimization report")
        
        try:
            report = {
                'timestamp': datetime.utcnow().isoformat(),
                'postgres_analysis': await self.query_analyzer.analyze_postgres_performance(),
                'postgres_optimizations': await self.query_analyzer.optimize_postgres_queries(),
                'redis_analysis': await self.query_analyzer.analyze_redis_performance(),
                'redis_optimizations': await self.query_analyzer.optimize_redis_cache(),
                'anomalies': await self.performance_monitor.detect_anomalies(),
                'system_metrics': await self.performance_monitor.collect_system_metrics()
            }
            
            # Store report in Redis for later retrieval
            report_key = f"optimization_report:{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            await self.db_connections.redis_client.setex(
                report_key,
                86400,  # 24 hours
                json.dumps(report, default=str)
            )
            
            logger.info("Optimization report generated", report_key=report_key)
            
        except Exception as e:
            logger.error("Report generation failed", error=str(e))
    
    async def run(self):
        """Run the optimizer service"""
        logger.info("Starting optimizer service")
        
        try:
            await self.initialize()
            
            # Keep the service running
            while True:
                await asyncio.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error("Service error", error=str(e))
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Shutdown the optimizer service"""
        logger.info("Shutting down optimizer service")
        
        if self.scheduler:
            self.scheduler.shutdown()
        
        await self.db_connections.close()

if __name__ == "__main__":
    service = OptimizerService()
    asyncio.run(service.run())