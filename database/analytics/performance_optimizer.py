#!/usr/bin/env python3
"""
Performance Optimization and Caching for Physics Assistant Analytics
Advanced caching strategies, query optimization, and performance tuning
for learning analytics calculations to ensure scalable real-time performance.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import hashlib
import weakref
import threading
from functools import wraps, lru_cache
import time
import pickle
import zlib
from concurrent.futures import ThreadPoolExecutor
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    data: Any
    timestamp: datetime
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[int] = None
    tags: List[str] = field(default_factory=list)

@dataclass
class PerformanceMetrics:
    """Performance metrics tracking"""
    cache_hits: int = 0
    cache_misses: int = 0
    total_requests: int = 0
    avg_response_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    optimization_savings_ms: float = 0.0

class LRUCache:
    """High-performance Least Recently Used cache with size limits"""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache = OrderedDict()
        self.current_memory = 0
        self.lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check TTL
                if entry.ttl_seconds:
                    age = (datetime.now() - entry.timestamp).total_seconds()
                    if age > entry.ttl_seconds:
                        self._remove_entry(key)
                        return None
                
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                entry.access_count += 1
                return entry.data
            
            return None
    
    def set(self, key: str, data: Any, ttl_seconds: Optional[int] = None, tags: List[str] = None):
        """Set item in cache"""
        with self.lock:
            # Calculate size
            size_bytes = self._calculate_size(data)
            
            # Remove existing entry if present
            if key in self.cache:
                self._remove_entry(key)
            
            # Ensure we have space
            self._ensure_space(size_bytes)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                data=data,
                timestamp=datetime.now(),
                size_bytes=size_bytes,
                ttl_seconds=ttl_seconds,
                tags=tags or []
            )
            
            # Add to cache
            self.cache[key] = entry
            self.current_memory += size_bytes
    
    def remove(self, key: str) -> bool:
        """Remove item from cache"""
        with self.lock:
            if key in self.cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear_by_tags(self, tags: List[str]):
        """Clear cache entries by tags"""
        with self.lock:
            keys_to_remove = []
            for key, entry in self.cache.items():
                if any(tag in entry.tags for tag in tags):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self._remove_entry(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'memory_usage_mb': self.current_memory / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'utilization': len(self.cache) / self.max_size,
                'memory_utilization': self.current_memory / self.max_memory_bytes
            }
    
    def _remove_entry(self, key: str):
        """Remove entry from cache"""
        if key in self.cache:
            entry = self.cache[key]
            self.current_memory -= entry.size_bytes
            del self.cache[key]
    
    def _ensure_space(self, needed_bytes: int):
        """Ensure cache has enough space"""
        # Remove expired entries first
        self._cleanup_expired()
        
        # Remove LRU entries if needed
        while (len(self.cache) >= self.max_size or 
               self.current_memory + needed_bytes > self.max_memory_bytes):
            if not self.cache:
                break
            
            # Remove least recently used item
            oldest_key = next(iter(self.cache))
            self._remove_entry(oldest_key)
    
    def _cleanup_expired(self):
        """Remove expired entries"""
        current_time = datetime.now()
        expired_keys = []
        
        for key, entry in self.cache.items():
            if entry.ttl_seconds:
                age = (current_time - entry.timestamp).total_seconds()
                if age > entry.ttl_seconds:
                    expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_entry(key)
    
    def _calculate_size(self, data: Any) -> int:
        """Calculate approximate size of data in bytes"""
        try:
            # Use pickle for size estimation
            return len(pickle.dumps(data))
        except:
            # Fallback estimation
            if isinstance(data, str):
                return len(data.encode('utf-8'))
            elif isinstance(data, (int, float)):
                return 8
            elif isinstance(data, dict):
                return sum(len(str(k)) + len(str(v)) for k, v in data.items())
            elif isinstance(data, list):
                return sum(len(str(item)) for item in data)
            else:
                return 1024  # Default estimate

class AnalyticsCache:
    """Specialized cache for analytics operations"""
    
    def __init__(self, redis_client=None, max_memory_mb: int = 500):
        self.redis_client = redis_client
        self.local_cache = LRUCache(max_size=2000, max_memory_mb=max_memory_mb)
        self.performance_metrics = PerformanceMetrics()
        
        # Cache configurations for different data types
        self.cache_configs = {
            'student_progress': {'ttl': 300, 'tags': ['student', 'progress']},  # 5 minutes
            'concept_mastery': {'ttl': 600, 'tags': ['student', 'mastery']},   # 10 minutes
            'learning_paths': {'ttl': 1800, 'tags': ['paths']},                # 30 minutes
            'educational_insights': {'ttl': 3600, 'tags': ['insights']},       # 1 hour
            'student_clusters': {'ttl': 7200, 'tags': ['clusters']},           # 2 hours
            'system_metrics': {'ttl': 60, 'tags': ['metrics']},                # 1 minute
        }
    
    async def get(self, key: str, data_type: str = 'default') -> Optional[Any]:
        """Get data from cache with fallback to Redis"""
        start_time = time.time()
        
        try:
            # Try local cache first
            data = self.local_cache.get(key)
            if data is not None:
                self.performance_metrics.cache_hits += 1
                return data
            
            # Try Redis cache if available
            if self.redis_client:
                try:
                    redis_data = await self.redis_client.get(f"analytics:{key}")
                    if redis_data:
                        # Decompress and deserialize
                        decompressed = zlib.decompress(redis_data.encode() if isinstance(redis_data, str) else redis_data)
                        data = pickle.loads(decompressed)
                        
                        # Store in local cache
                        config = self.cache_configs.get(data_type, {})
                        self.local_cache.set(key, data, config.get('ttl'), config.get('tags'))
                        
                        self.performance_metrics.cache_hits += 1
                        return data
                except Exception as e:
                    logger.warning(f"Redis cache error: {e}")
            
            self.performance_metrics.cache_misses += 1
            return None
            
        finally:
            response_time = (time.time() - start_time) * 1000
            self._update_response_time(response_time)
    
    async def set(self, key: str, data: Any, data_type: str = 'default'):
        """Set data in cache with Redis backup"""
        try:
            config = self.cache_configs.get(data_type, {})
            ttl = config.get('ttl')
            tags = config.get('tags', [])
            
            # Store in local cache
            self.local_cache.set(key, data, ttl, tags)
            
            # Store in Redis if available
            if self.redis_client and ttl:
                try:
                    # Serialize and compress
                    serialized = pickle.dumps(data)
                    compressed = zlib.compress(serialized)
                    
                    await self.redis_client.set(f"analytics:{key}", compressed, ex=ttl)
                except Exception as e:
                    logger.warning(f"Redis cache write error: {e}")
        
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    async def invalidate(self, pattern: str = None, tags: List[str] = None):
        """Invalidate cache entries"""
        try:
            if tags:
                self.local_cache.clear_by_tags(tags)
                
                # Clear from Redis if available
                if self.redis_client:
                    try:
                        # This would require a more sophisticated Redis key pattern matching
                        pass
                    except Exception as e:
                        logger.warning(f"Redis cache invalidation error: {e}")
        
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        # Update system metrics
        self.performance_metrics.memory_usage_mb = psutil.virtual_memory().used / (1024 * 1024)
        self.performance_metrics.cpu_usage_percent = psutil.cpu_percent()
        
        # Calculate cache hit rate
        total_requests = self.performance_metrics.cache_hits + self.performance_metrics.cache_misses
        if total_requests > 0:
            hit_rate = self.performance_metrics.cache_hits / total_requests
            self.performance_metrics.total_requests = total_requests
        
        return self.performance_metrics
    
    def _update_response_time(self, response_time_ms: float):
        """Update average response time"""
        current_avg = self.performance_metrics.avg_response_time
        total_requests = self.performance_metrics.cache_hits + self.performance_metrics.cache_misses
        
        if total_requests > 1:
            self.performance_metrics.avg_response_time = (
                (current_avg * (total_requests - 1) + response_time_ms) / total_requests
            )
        else:
            self.performance_metrics.avg_response_time = response_time_ms

class QueryOptimizer:
    """Database query optimization for analytics"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.query_cache = {}
        self.query_stats = defaultdict(lambda: {'count': 0, 'total_time': 0})
        
    async def execute_optimized_query(self, query: str, params: tuple = (), 
                                    cache_key: str = None, ttl: int = 300) -> List[Dict]:
        """Execute query with optimization and caching"""
        start_time = time.time()
        
        try:
            # Generate cache key if not provided
            if not cache_key:
                cache_key = self._generate_query_key(query, params)
            
            # Check cache first
            if cache_key in self.query_cache:
                cache_entry = self.query_cache[cache_key]
                if (datetime.now() - cache_entry['timestamp']).total_seconds() < ttl:
                    return cache_entry['data']
                else:
                    del self.query_cache[cache_key]
            
            # Execute query
            if self.db_manager:
                async with self.db_manager.postgres.get_connection() as conn:
                    # Optimize query if possible
                    optimized_query = self._optimize_query(query)
                    
                    results = await conn.fetch(optimized_query, *params)
                    
                    # Convert to list of dicts
                    data = [dict(row) for row in results]
                    
                    # Cache results
                    self.query_cache[cache_key] = {
                        'data': data,
                        'timestamp': datetime.now()
                    }
                    
                    return data
            else:
                return []
        
        finally:
            execution_time = time.time() - start_time
            self.query_stats[query]['count'] += 1
            self.query_stats[query]['total_time'] += execution_time
    
    def _generate_query_key(self, query: str, params: tuple) -> str:
        """Generate cache key for query"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        params_hash = hashlib.md5(str(params).encode()).hexdigest()
        return f"query:{query_hash}:{params_hash}"
    
    def _optimize_query(self, query: str) -> str:
        """Apply query optimizations"""
        optimized_query = query
        
        # Add common optimizations
        if 'ORDER BY created_at' in query and 'LIMIT' in query:
            # Ensure we have proper index usage
            optimized_query = query.replace(
                'ORDER BY created_at',
                'ORDER BY created_at DESC'
            )
        
        # Add other optimizations as needed
        return optimized_query
    
    def get_query_stats(self) -> Dict[str, Any]:
        """Get query performance statistics"""
        stats = {}
        for query, data in self.query_stats.items():
            if data['count'] > 0:
                stats[query[:100]] = {  # Truncate long queries
                    'executions': data['count'],
                    'total_time': data['total_time'],
                    'avg_time': data['total_time'] / data['count']
                }
        
        return stats

class BatchProcessor:
    """Batch processing for analytics calculations"""
    
    def __init__(self, max_batch_size: int = 50, max_workers: int = 4):
        self.max_batch_size = max_batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.pending_operations = defaultdict(list)
        self.batch_timers = {}
        
    async def add_operation(self, operation_type: str, operation_data: Dict, 
                          callback: Callable = None, delay_ms: int = 100):
        """Add operation to batch queue"""
        operation = {
            'data': operation_data,
            'callback': callback,
            'timestamp': datetime.now()
        }
        
        self.pending_operations[operation_type].append(operation)
        
        # Start batch timer if not already running
        if operation_type not in self.batch_timers:
            self.batch_timers[operation_type] = asyncio.create_task(
                self._process_batch_after_delay(operation_type, delay_ms)
            )
        
        # Process immediately if batch is full
        if len(self.pending_operations[operation_type]) >= self.max_batch_size:
            await self._process_batch(operation_type)
    
    async def _process_batch_after_delay(self, operation_type: str, delay_ms: int):
        """Process batch after specified delay"""
        await asyncio.sleep(delay_ms / 1000.0)
        await self._process_batch(operation_type)
    
    async def _process_batch(self, operation_type: str):
        """Process pending batch operations"""
        if operation_type not in self.pending_operations:
            return
        
        operations = self.pending_operations[operation_type]
        if not operations:
            return
        
        # Clear pending operations
        self.pending_operations[operation_type] = []
        
        # Cancel timer
        if operation_type in self.batch_timers:
            self.batch_timers[operation_type].cancel()
            del self.batch_timers[operation_type]
        
        try:
            # Process operations in batch
            if operation_type == 'student_progress_update':
                await self._process_progress_batch(operations)
            elif operation_type == 'concept_mastery_calculation':
                await self._process_mastery_batch(operations)
            elif operation_type == 'analytics_event_processing':
                await self._process_event_batch(operations)
            
        except Exception as e:
            logger.error(f"Batch processing error for {operation_type}: {e}")
    
    async def _process_progress_batch(self, operations: List[Dict]):
        """Process student progress updates in batch"""
        try:
            # Group by user_id for efficient processing
            user_operations = defaultdict(list)
            for op in operations:
                user_id = op['data'].get('user_id')
                if user_id:
                    user_operations[user_id].append(op)
            
            # Process each user's operations
            for user_id, user_ops in user_operations.items():
                # Combine operations for this user
                combined_data = self._combine_progress_operations(user_ops)
                
                # Execute callbacks
                for op in user_ops:
                    if op['callback']:
                        try:
                            await op['callback'](combined_data)
                        except Exception as e:
                            logger.error(f"Progress callback error: {e}")
        
        except Exception as e:
            logger.error(f"Progress batch processing error: {e}")
    
    async def _process_mastery_batch(self, operations: List[Dict]):
        """Process concept mastery calculations in batch"""
        try:
            # Group by concept for efficient processing
            concept_operations = defaultdict(list)
            for op in operations:
                concept = op['data'].get('concept')
                if concept:
                    concept_operations[concept].append(op)
            
            # Process each concept's operations
            for concept, concept_ops in concept_operations.items():
                # Process operations for this concept
                for op in concept_ops:
                    if op['callback']:
                        try:
                            await op['callback'](op['data'])
                        except Exception as e:
                            logger.error(f"Mastery callback error: {e}")
        
        except Exception as e:
            logger.error(f"Mastery batch processing error: {e}")
    
    async def _process_event_batch(self, operations: List[Dict]):
        """Process analytics events in batch"""
        try:
            # Sort by timestamp for chronological processing
            operations.sort(key=lambda x: x['timestamp'])
            
            # Process events
            for op in operations:
                if op['callback']:
                    try:
                        await op['callback'](op['data'])
                    except Exception as e:
                        logger.error(f"Event callback error: {e}")
        
        except Exception as e:
            logger.error(f"Event batch processing error: {e}")
    
    def _combine_progress_operations(self, operations: List[Dict]) -> Dict:
        """Combine multiple progress operations for the same user"""
        combined = {
            'user_id': operations[0]['data'].get('user_id'),
            'updates': [],
            'timestamp': datetime.now()
        }
        
        for op in operations:
            combined['updates'].append(op['data'])
        
        return combined

class PerformanceMonitor:
    """Real-time performance monitoring for analytics"""
    
    def __init__(self):
        self.metrics_history = defaultdict(list)
        self.alerts = []
        self.thresholds = {
            'response_time_ms': 1000,
            'memory_usage_mb': 1000,
            'cpu_usage_percent': 80,
            'cache_hit_rate': 0.7,
            'error_rate': 0.05
        }
    
    def record_metric(self, metric_name: str, value: float, timestamp: datetime = None):
        """Record a performance metric"""
        if not timestamp:
            timestamp = datetime.now()
        
        self.metrics_history[metric_name].append({
            'value': value,
            'timestamp': timestamp
        })
        
        # Keep only recent metrics (last hour)
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.metrics_history[metric_name] = [
            m for m in self.metrics_history[metric_name] 
            if m['timestamp'] > cutoff_time
        ]
        
        # Check thresholds
        self._check_threshold(metric_name, value)
    
    def _check_threshold(self, metric_name: str, value: float):
        """Check if metric exceeds threshold"""
        if metric_name in self.thresholds:
            threshold = self.thresholds[metric_name]
            
            if metric_name == 'cache_hit_rate' and value < threshold:
                self._create_alert(f"Low cache hit rate: {value:.2f} < {threshold}")
            elif metric_name != 'cache_hit_rate' and value > threshold:
                self._create_alert(f"High {metric_name}: {value:.2f} > {threshold}")
    
    def _create_alert(self, message: str):
        """Create performance alert"""
        alert = {
            'message': message,
            'timestamp': datetime.now(),
            'severity': 'warning'
        }
        
        self.alerts.append(alert)
        
        # Keep only recent alerts
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.alerts = [a for a in self.alerts if a['timestamp'] > cutoff_time]
        
        logger.warning(f"Performance Alert: {message}")
    
    def get_metrics_summary(self, window_minutes: int = 30) -> Dict[str, Any]:
        """Get metrics summary for the specified time window"""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        summary = {}
        
        for metric_name, history in self.metrics_history.items():
            recent_values = [
                m['value'] for m in history 
                if m['timestamp'] > cutoff_time
            ]
            
            if recent_values:
                summary[metric_name] = {
                    'current': recent_values[-1],
                    'average': np.mean(recent_values),
                    'max': max(recent_values),
                    'min': min(recent_values),
                    'count': len(recent_values)
                }
        
        return summary
    
    def get_recent_alerts(self, hours: int = 1) -> List[Dict]:
        """Get recent performance alerts"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [a for a in self.alerts if a['timestamp'] > cutoff_time]

# Decorators for performance optimization
def cached_method(ttl: int = 300, data_type: str = 'default'):
    """Decorator for caching method results"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            if hasattr(self, '_cache'):
                cached_result = await self._cache.get(cache_key, data_type)
                if cached_result is not None:
                    return cached_result
            
            # Execute function
            result = await func(self, *args, **kwargs)
            
            # Cache result
            if hasattr(self, '_cache') and result is not None:
                await self._cache.set(cache_key, result, data_type)
            
            return result
        
        return wrapper
    return decorator

def batch_operation(operation_type: str, delay_ms: int = 100):
    """Decorator for batching operations"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Add to batch processor if available
            if hasattr(self, '_batch_processor'):
                operation_data = {
                    'function': func.__name__,
                    'args': args,
                    'kwargs': kwargs
                }
                
                async def callback(data):
                    return await func(self, *args, **kwargs)
                
                await self._batch_processor.add_operation(
                    operation_type, operation_data, callback, delay_ms
                )
                return None
            else:
                # Execute immediately if no batch processor
                return await func(self, *args, **kwargs)
        
        return wrapper
    return decorator

def monitor_performance(metric_name: str):
    """Decorator for monitoring function performance"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Record successful execution time
                execution_time = (time.time() - start_time) * 1000
                
                # Record metric if monitor available
                if hasattr(args[0], '_performance_monitor'):
                    args[0]._performance_monitor.record_metric(
                        f"{metric_name}_response_time_ms", execution_time
                    )
                
                return result
            
            except Exception as e:
                # Record error
                if hasattr(args[0], '_performance_monitor'):
                    args[0]._performance_monitor.record_metric(
                        f"{metric_name}_errors", 1
                    )
                raise
        
        return wrapper
    return decorator

# Main performance optimization manager
class AnalyticsPerformanceManager:
    """Central manager for analytics performance optimization"""
    
    def __init__(self, db_manager=None, redis_client=None):
        self.cache = AnalyticsCache(redis_client, max_memory_mb=500)
        self.query_optimizer = QueryOptimizer(db_manager)
        self.batch_processor = BatchProcessor(max_batch_size=50, max_workers=4)
        self.performance_monitor = PerformanceMonitor()
        
    async def optimize_analytics_engine(self, analytics_engine):
        """Add performance optimizations to analytics engine"""
        analytics_engine._cache = self.cache
        analytics_engine._batch_processor = self.batch_processor
        analytics_engine._performance_monitor = self.performance_monitor
        analytics_engine._query_optimizer = self.query_optimizer
        
        logger.info("✅ Analytics engine optimized with caching and performance monitoring")
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        cache_stats = self.cache.local_cache.get_stats()
        cache_metrics = self.cache.get_performance_metrics()
        query_stats = self.query_optimizer.get_query_stats()
        metrics_summary = self.performance_monitor.get_metrics_summary()
        recent_alerts = self.performance_monitor.get_recent_alerts()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cache_statistics': cache_stats,
            'cache_performance': {
                'hit_rate': cache_metrics.cache_hits / max(cache_metrics.total_requests, 1),
                'avg_response_time_ms': cache_metrics.avg_response_time,
                'memory_usage_mb': cache_metrics.memory_usage_mb,
                'cpu_usage_percent': cache_metrics.cpu_usage_percent
            },
            'query_performance': query_stats,
            'system_metrics': metrics_summary,
            'recent_alerts': recent_alerts,
            'optimization_recommendations': self._generate_optimization_recommendations(
                cache_stats, cache_metrics, metrics_summary
            )
        }
    
    def _generate_optimization_recommendations(self, cache_stats: Dict, 
                                             cache_metrics: PerformanceMetrics,
                                             metrics_summary: Dict) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Cache recommendations
        if cache_stats['utilization'] > 0.9:
            recommendations.append("Consider increasing cache size")
        
        hit_rate = cache_metrics.cache_hits / max(cache_metrics.total_requests, 1)
        if hit_rate < 0.7:
            recommendations.append("Low cache hit rate - review caching strategy")
        
        if cache_metrics.avg_response_time > 100:
            recommendations.append("High average response time - consider query optimization")
        
        # Memory recommendations
        if cache_metrics.memory_usage_mb > 800:
            recommendations.append("High memory usage - consider memory optimization")
        
        # CPU recommendations
        if cache_metrics.cpu_usage_percent > 80:
            recommendations.append("High CPU usage - consider load balancing")
        
        return recommendations

# Example usage and testing
async def test_performance_optimization():
    """Test performance optimization components"""
    try:
        logger.info("🧪 Testing Performance Optimization")
        
        # Test LRU Cache
        cache = LRUCache(max_size=100, max_memory_mb=10)
        
        # Test cache operations
        cache.set("test_key", {"data": "test_value"}, ttl_seconds=60)
        result = cache.get("test_key")
        assert result == {"data": "test_value"}
        
        # Test cache stats
        stats = cache.get_stats()
        assert stats['size'] == 1
        
        logger.info("✅ LRU Cache test passed")
        
        # Test Analytics Cache
        analytics_cache = AnalyticsCache(max_memory_mb=50)
        
        await analytics_cache.set("analytics_key", {"analytics": "data"}, "student_progress")
        result = await analytics_cache.get("analytics_key", "student_progress")
        assert result == {"analytics": "data"}
        
        logger.info("✅ Analytics Cache test passed")
        
        # Test Query Optimizer
        query_optimizer = QueryOptimizer()
        
        # Test query optimization
        optimized = query_optimizer._optimize_query(
            "SELECT * FROM interactions ORDER BY created_at LIMIT 10"
        )
        assert "ORDER BY created_at DESC" in optimized
        
        logger.info("✅ Query Optimizer test passed")
        
        # Test Batch Processor
        batch_processor = BatchProcessor(max_batch_size=5, max_workers=2)
        
        # Test batch operation
        processed_operations = []
        
        async def test_callback(data):
            processed_operations.append(data)
        
        await batch_processor.add_operation(
            "test_operation", 
            {"test": "data"}, 
            test_callback,
            delay_ms=10
        )
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        logger.info("✅ Batch Processor test passed")
        
        # Test Performance Monitor
        monitor = PerformanceMonitor()
        
        # Record test metrics
        monitor.record_metric("test_metric", 50.0)
        monitor.record_metric("test_metric", 75.0)
        
        summary = monitor.get_metrics_summary(window_minutes=1)
        assert "test_metric" in summary
        
        logger.info("✅ Performance Monitor test passed")
        
        logger.info("🎉 All performance optimization tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Performance optimization test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(test_performance_optimization())