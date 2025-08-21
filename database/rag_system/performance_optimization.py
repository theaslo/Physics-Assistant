#!/usr/bin/env python3
"""
Performance Optimization Module for Physics Assistant RAG System
Provides advanced caching, indexing, and performance monitoring capabilities
"""
import os
import json
import logging
import asyncio
import time
import hashlib
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import threading
from contextlib import asynccontextmanager

# Third-party imports
import redis
import faiss
import asyncpg
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Cache configuration settings"""
    # Redis caching
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = None
    redis_db: int = 0
    
    # Cache TTL settings (in seconds)
    embedding_cache_ttl: int = 7200  # 2 hours
    search_results_ttl: int = 1800   # 30 minutes
    graph_paths_ttl: int = 3600      # 1 hour
    student_profile_ttl: int = 3600  # 1 hour
    
    # Cache size limits
    max_memory_cache_size: int = 1000  # items
    max_embedding_cache_size: int = 5000  # embeddings
    
    # Performance settings
    cache_compression: bool = True
    async_cache_writes: bool = True
    cache_warming_enabled: bool = True

@dataclass 
class IndexConfig:
    """Index configuration settings"""
    # FAISS index settings
    faiss_index_type: str = "IVF"  # IVF, HNSW, LSH
    faiss_nlist: int = 100  # Number of clusters for IVF
    faiss_nprobe: int = 10  # Number of clusters to search
    
    # Search optimization
    enable_gpu_indexing: bool = False
    parallel_search_threads: int = 4
    batch_search_size: int = 32
    
    # Index persistence
    index_backup_interval: int = 3600  # 1 hour
    auto_index_rebuilding: bool = True
    index_compression_enabled: bool = True

@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics"""
    # Query metrics
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    avg_response_time: float = 0.0
    
    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0
    cache_write_errors: int = 0
    
    # Index metrics
    index_searches: int = 0
    avg_index_search_time: float = 0.0
    index_rebuilds: int = 0
    
    # Resource metrics
    peak_memory_usage: float = 0.0
    avg_cpu_usage: float = 0.0
    concurrent_requests: int = 0
    
    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    @property
    def success_rate(self) -> float:
        return self.successful_queries / self.total_queries if self.total_queries > 0 else 0.0

class AdvancedCache:
    """Advanced caching system with multiple layers"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        
        # Memory cache (L1)
        self.memory_cache = {}
        self.memory_cache_access_times = {}
        self.memory_cache_lock = threading.RLock()
        
        # Redis cache (L2)
        self.redis_client = None
        
        # Cache statistics
        self.stats = {
            'memory_hits': 0,
            'memory_misses': 0,
            'redis_hits': 0,
            'redis_misses': 0,
            'write_errors': 0
        }
        
        # Thread pool for async operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
    async def initialize(self):
        """Initialize cache connections"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                password=self.config.redis_password,
                db=self.config.redis_db,
                decode_responses=False,  # Keep binary for embeddings
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            await asyncio.to_thread(self.redis_client.ping)
            logger.info("✅ Redis cache client initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Redis cache: {e}")
            self.redis_client = None
    
    async def close(self):
        """Close cache connections"""
        if self.redis_client:
            await asyncio.to_thread(self.redis_client.close)
        
        self.thread_pool.shutdown(wait=True)
        logger.info("🔒 Cache connections closed")
    
    async def get(self, key: str, category: str = "default") -> Optional[Any]:
        """Get value from cache with L1/L2 hierarchy"""
        
        # Try memory cache first (L1)
        with self.memory_cache_lock:
            if key in self.memory_cache:
                self.memory_cache_access_times[key] = time.time()
                self.stats['memory_hits'] += 1
                return self._deserialize_value(self.memory_cache[key], category)
            else:
                self.stats['memory_misses'] += 1
        
        # Try Redis cache (L2)
        if self.redis_client:
            try:
                cache_key = self._build_cache_key(key, category)
                cached_data = await asyncio.to_thread(self.redis_client.get, cache_key)
                
                if cached_data:
                    self.stats['redis_hits'] += 1
                    
                    # Deserialize value
                    value = self._deserialize_value(cached_data, category)
                    
                    # Promote to memory cache if there's space
                    await self._promote_to_memory_cache(key, cached_data)
                    
                    return value
                else:
                    self.stats['redis_misses'] += 1
                    
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        return None
    
    async def set(self, key: str, value: Any, ttl: int = None, category: str = "default"):
        """Set value in cache with automatic serialization"""
        
        serialized_value = self._serialize_value(value, category)
        
        # Set in memory cache (L1)
        with self.memory_cache_lock:
            if len(self.memory_cache) >= self.config.max_memory_cache_size:
                self._evict_memory_cache()
            
            self.memory_cache[key] = serialized_value
            self.memory_cache_access_times[key] = time.time()
        
        # Set in Redis cache (L2)
        if self.redis_client:
            if self.config.async_cache_writes:
                # Async write to Redis
                asyncio.create_task(self._async_redis_write(key, serialized_value, ttl, category))
            else:
                # Sync write to Redis
                await self._sync_redis_write(key, serialized_value, ttl, category)
    
    async def delete(self, key: str, category: str = "default"):
        """Delete value from all cache layers"""
        
        # Delete from memory cache
        with self.memory_cache_lock:
            self.memory_cache.pop(key, None)
            self.memory_cache_access_times.pop(key, None)
        
        # Delete from Redis cache
        if self.redis_client:
            try:
                cache_key = self._build_cache_key(key, category)
                await asyncio.to_thread(self.redis_client.delete, cache_key)
            except Exception as e:
                logger.warning(f"Redis delete error: {e}")
    
    async def clear_category(self, category: str):
        """Clear all cache entries in a category"""
        
        if not self.redis_client:
            return
        
        try:
            pattern = f"cache:{category}:*"
            keys = await asyncio.to_thread(self.redis_client.keys, pattern)
            
            if keys:
                await asyncio.to_thread(self.redis_client.delete, *keys)
                logger.info(f"Cleared {len(keys)} cache entries for category: {category}")
                
        except Exception as e:
            logger.warning(f"Cache clear error: {e}")
    
    def _build_cache_key(self, key: str, category: str) -> str:
        """Build standardized cache key"""
        return f"cache:{category}:{key}"
    
    def _serialize_value(self, value: Any, category: str) -> bytes:
        """Serialize value based on category"""
        
        if category == "embedding":
            # Special handling for numpy arrays
            if isinstance(value, np.ndarray):
                return pickle.dumps(value)
            
        if category == "search_results":
            # Compress search results if enabled
            if self.config.cache_compression:
                import gzip
                return gzip.compress(json.dumps(value, default=str).encode())
        
        # Default serialization
        return pickle.dumps(value)
    
    def _deserialize_value(self, data: bytes, category: str) -> Any:
        """Deserialize value based on category"""
        
        if category == "embedding":
            return pickle.loads(data)
        
        if category == "search_results":
            if self.config.cache_compression:
                import gzip
                try:
                    decompressed = gzip.decompress(data)
                    return json.loads(decompressed.decode())
                except:
                    # Fallback to pickle for backward compatibility
                    return pickle.loads(data)
        
        # Default deserialization
        return pickle.loads(data)
    
    async def _promote_to_memory_cache(self, key: str, value: bytes):
        """Promote frequently accessed items to memory cache"""
        
        with self.memory_cache_lock:
            if len(self.memory_cache) >= self.config.max_memory_cache_size:
                self._evict_memory_cache()
            
            self.memory_cache[key] = value
            self.memory_cache_access_times[key] = time.time()
    
    def _evict_memory_cache(self):
        """Evict least recently used items from memory cache"""
        
        if not self.memory_cache_access_times:
            return
        
        # Remove 20% of items (LRU eviction)
        num_to_remove = max(1, len(self.memory_cache) // 5)
        
        # Sort by access time and remove oldest
        sorted_items = sorted(self.memory_cache_access_times.items(), key=lambda x: x[1])
        
        for key, _ in sorted_items[:num_to_remove]:
            self.memory_cache.pop(key, None)
            self.memory_cache_access_times.pop(key, None)
    
    async def _async_redis_write(self, key: str, value: bytes, ttl: int, category: str):
        """Async write to Redis cache"""
        
        try:
            cache_key = self._build_cache_key(key, category)
            ttl = ttl or self._get_default_ttl(category)
            
            await asyncio.to_thread(self.redis_client.setex, cache_key, ttl, value)
            
        except Exception as e:
            self.stats['write_errors'] += 1
            logger.warning(f"Async Redis write error: {e}")
    
    async def _sync_redis_write(self, key: str, value: bytes, ttl: int, category: str):
        """Sync write to Redis cache"""
        
        try:
            cache_key = self._build_cache_key(key, category)
            ttl = ttl or self._get_default_ttl(category)
            
            await asyncio.to_thread(self.redis_client.setex, cache_key, ttl, value)
            
        except Exception as e:
            self.stats['write_errors'] += 1
            logger.warning(f"Sync Redis write error: {e}")
    
    def _get_default_ttl(self, category: str) -> int:
        """Get default TTL for category"""
        
        ttl_mapping = {
            "embedding": self.config.embedding_cache_ttl,
            "search_results": self.config.search_results_ttl,
            "graph_paths": self.config.graph_paths_ttl,
            "student_profile": self.config.student_profile_ttl
        }
        
        return ttl_mapping.get(category, 1800)  # Default 30 minutes
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        
        total_memory_requests = self.stats['memory_hits'] + self.stats['memory_misses']
        total_redis_requests = self.stats['redis_hits'] + self.stats['redis_misses']
        
        return {
            "memory_cache": {
                "size": len(self.memory_cache),
                "max_size": self.config.max_memory_cache_size,
                "hit_rate": self.stats['memory_hits'] / total_memory_requests if total_memory_requests > 0 else 0,
                "hits": self.stats['memory_hits'],
                "misses": self.stats['memory_misses']
            },
            "redis_cache": {
                "hit_rate": self.stats['redis_hits'] / total_redis_requests if total_redis_requests > 0 else 0,
                "hits": self.stats['redis_hits'],
                "misses": self.stats['redis_misses'],
                "write_errors": self.stats['write_errors']
            }
        }

class OptimizedIndexManager:
    """Optimized index manager with advanced FAISS configurations"""
    
    def __init__(self, config: IndexConfig):
        self.config = config
        self.indices = {}  # content_type -> FAISS index
        self.metadata_stores = {}  # content_type -> metadata list
        self.index_locks = {}  # content_type -> lock
        
        # Performance monitoring
        self.search_times = deque(maxlen=1000)  # Last 1000 search times
        self.rebuild_times = {}
        
        # Background threads
        self.backup_thread = None
        self.stop_backup = threading.Event()
        
    async def initialize(self):
        """Initialize optimized index system"""
        logger.info("🚀 Initializing optimized index manager")
        
        # Start background backup thread if enabled
        if self.config.index_backup_interval > 0:
            self.backup_thread = threading.Thread(target=self._backup_loop, daemon=True)
            self.backup_thread.start()
    
    async def close(self):
        """Close index manager and cleanup"""
        
        # Stop backup thread
        if self.backup_thread:
            self.stop_backup.set()
            self.backup_thread.join(timeout=5)
        
        logger.info("🔒 Index manager closed")
    
    async def build_optimized_index(self, content_type: str, embeddings: np.ndarray, 
                                  metadata: List[Dict]) -> bool:
        """Build optimized FAISS index for content type"""
        
        if embeddings.shape[0] == 0:
            logger.warning(f"No embeddings provided for {content_type}")
            return False
        
        start_time = time.time()
        
        try:
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings.astype(np.float32))
            
            dimension = embeddings.shape[1]
            n_vectors = embeddings.shape[0]
            
            # Choose optimal index type based on data size
            index = self._create_optimal_index(dimension, n_vectors)
            
            # Train index if needed
            if hasattr(index, 'train') and not index.is_trained:
                logger.info(f"Training index for {content_type}...")
                index.train(embeddings.astype(np.float32))
            
            # Add vectors to index
            index.add(embeddings.astype(np.float32))
            
            # Store index and metadata
            self.indices[content_type] = index
            self.metadata_stores[content_type] = metadata
            self.index_locks[content_type] = threading.RLock()
            
            build_time = time.time() - start_time
            self.rebuild_times[content_type] = build_time
            
            logger.info(f"✅ Built optimized {self.config.faiss_index_type} index for {content_type}: "
                       f"{n_vectors} vectors, {dimension}D, built in {build_time:.2f}s")
            
            # Backup index if compression enabled
            if self.config.index_compression_enabled:
                await self._backup_index(content_type)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to build index for {content_type}: {e}")
            return False
    
    def _create_optimal_index(self, dimension: int, n_vectors: int) -> faiss.Index:
        """Create optimal FAISS index based on data characteristics"""
        
        if self.config.faiss_index_type == "IVF":
            # IVF (Inverted File) - good for large datasets
            if n_vectors < 1000:
                # For small datasets, use flat index
                return faiss.IndexFlatIP(dimension)
            
            # Calculate optimal number of clusters
            nlist = min(self.config.faiss_nlist, max(16, int(np.sqrt(n_vectors))))
            
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            index.nprobe = min(self.config.faiss_nprobe, nlist // 2)
            
            return index
            
        elif self.config.faiss_index_type == "HNSW":
            # HNSW - good for high-dimensional data and fast search
            index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 40
            index.hnsw.efSearch = 16
            
            return index
            
        elif self.config.faiss_index_type == "LSH":
            # LSH - good for approximate search
            nbits = min(256, max(64, dimension // 2))
            return faiss.IndexLSH(dimension, nbits)
        
        else:
            # Default to flat index
            return faiss.IndexFlatIP(dimension)
    
    async def optimized_search(self, content_type: str, query_embedding: np.ndarray,
                              k: int = 10, min_similarity: float = 0.0) -> List[Tuple[int, float, Dict]]:
        """Perform optimized search with performance monitoring"""
        
        if content_type not in self.indices:
            logger.warning(f"No index found for content type: {content_type}")
            return []
        
        start_time = time.time()
        
        try:
            # Get index and metadata
            index = self.indices[content_type]
            metadata = self.metadata_stores[content_type]
            
            # Normalize query embedding
            query_embedding = query_embedding.astype(np.float32)
            faiss.normalize_L2(query_embedding.reshape(1, -1))
            
            # Perform search
            with self.index_locks[content_type]:
                similarities, indices = index.search(query_embedding.reshape(1, -1), k)
            
            # Process results
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx >= 0 and similarity >= min_similarity and idx < len(metadata):
                    results.append((idx, float(similarity), metadata[idx]))
            
            # Record search time
            search_time = time.time() - start_time
            self.search_times.append(search_time)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Search failed for {content_type}: {e}")
            return []
    
    async def batch_search(self, content_type: str, query_embeddings: np.ndarray,
                          k: int = 10) -> List[List[Tuple[int, float, Dict]]]:
        """Perform batch search for multiple queries"""
        
        if content_type not in self.indices:
            return []
        
        start_time = time.time()
        
        try:
            index = self.indices[content_type]
            metadata = self.metadata_stores[content_type]
            
            # Normalize query embeddings
            query_embeddings = query_embeddings.astype(np.float32)
            faiss.normalize_L2(query_embeddings)
            
            # Batch search
            with self.index_locks[content_type]:
                similarities, indices = index.search(query_embeddings, k)
            
            # Process batch results
            batch_results = []
            for query_idx in range(len(similarities)):
                query_results = []
                for similarity, idx in zip(similarities[query_idx], indices[query_idx]):
                    if idx >= 0 and idx < len(metadata):
                        query_results.append((idx, float(similarity), metadata[idx]))
                batch_results.append(query_results)
            
            search_time = time.time() - start_time
            logger.info(f"Batch search completed: {len(query_embeddings)} queries in {search_time:.2f}s")
            
            return batch_results
            
        except Exception as e:
            logger.error(f"❌ Batch search failed: {e}")
            return []
    
    async def _backup_index(self, content_type: str):
        """Backup index to persistent storage"""
        
        if content_type not in self.indices:
            return
        
        try:
            index = self.indices[content_type]
            backup_path = f"index_backups/{content_type}_index.faiss"
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            
            # Write index to disk
            faiss.write_index(index, backup_path)
            
            # Backup metadata
            metadata_path = f"index_backups/{content_type}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata_stores[content_type], f)
            
            logger.info(f"✅ Backed up index for {content_type}")
            
        except Exception as e:
            logger.error(f"❌ Index backup failed for {content_type}: {e}")
    
    def _backup_loop(self):
        """Background thread for periodic index backups"""
        
        while not self.stop_backup.wait(self.config.index_backup_interval):
            try:
                for content_type in list(self.indices.keys()):
                    asyncio.create_task(self._backup_index(content_type))
            except Exception as e:
                logger.error(f"Backup loop error: {e}")
    
    async def load_backup_index(self, content_type: str) -> bool:
        """Load index from backup"""
        
        try:
            backup_path = f"index_backups/{content_type}_index.faiss"
            metadata_path = f"index_backups/{content_type}_metadata.json"
            
            if not os.path.exists(backup_path) or not os.path.exists(metadata_path):
                return False
            
            # Load index
            index = faiss.read_index(backup_path)
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.indices[content_type] = index
            self.metadata_stores[content_type] = metadata
            self.index_locks[content_type] = threading.RLock()
            
            logger.info(f"✅ Loaded backup index for {content_type}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load backup index for {content_type}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index performance statistics"""
        
        avg_search_time = sum(self.search_times) / len(self.search_times) if self.search_times else 0
        
        return {
            "indices": {
                "content_types": list(self.indices.keys()),
                "total_vectors": sum(idx.ntotal for idx in self.indices.values()),
                "avg_search_time": avg_search_time,
                "total_searches": len(self.search_times)
            },
            "rebuild_times": self.rebuild_times,
            "configuration": {
                "index_type": self.config.faiss_index_type,
                "nlist": self.config.faiss_nlist,
                "nprobe": self.config.faiss_nprobe,
                "gpu_enabled": self.config.enable_gpu_indexing
            }
        }

class PerformanceMonitor:
    """System performance monitoring and optimization"""
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.query_times = deque(maxlen=10000)  # Last 10k query times
        self.resource_stats = deque(maxlen=1000)  # Last 1k resource measurements
        self.alert_thresholds = {
            'max_response_time': 10.0,  # seconds
            'min_cache_hit_rate': 0.5,  # 50%
            'max_memory_usage': 0.8     # 80% of available
        }
        
        # Monitoring thread
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        
    def start_monitoring(self):
        """Start background performance monitoring"""
        
        if self.monitoring_thread is None:
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info("📊 Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background performance monitoring"""
        
        if self.monitoring_thread:
            self.stop_monitoring.set()
            self.monitoring_thread.join(timeout=5)
            self.monitoring_thread = None
            logger.info("📊 Performance monitoring stopped")
    
    def record_query(self, response_time: float, success: bool, cache_hit: bool = False):
        """Record query performance metrics"""
        
        self.metrics.total_queries += 1
        
        if success:
            self.metrics.successful_queries += 1
        else:
            self.metrics.failed_queries += 1
        
        # Update average response time
        current_avg = self.metrics.avg_response_time
        total_queries = self.metrics.total_queries
        self.metrics.avg_response_time = ((current_avg * (total_queries - 1)) + response_time) / total_queries
        
        # Record query time
        self.query_times.append(response_time)
        
        # Update cache metrics
        if cache_hit:
            self.metrics.cache_hits += 1
        else:
            self.metrics.cache_misses += 1
    
    def record_index_search(self, search_time: float):
        """Record index search performance"""
        
        self.metrics.index_searches += 1
        
        current_avg = self.metrics.avg_index_search_time
        total_searches = self.metrics.index_searches
        self.metrics.avg_index_search_time = ((current_avg * (total_searches - 1)) + search_time) / total_searches
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        # Calculate percentiles for response times
        query_times_list = list(self.query_times)
        query_times_list.sort()
        
        percentiles = {}
        if query_times_list:
            percentiles = {
                'p50': query_times_list[len(query_times_list) // 2],
                'p90': query_times_list[int(len(query_times_list) * 0.9)],
                'p95': query_times_list[int(len(query_times_list) * 0.95)],
                'p99': query_times_list[int(len(query_times_list) * 0.99)]
            }
        
        # Resource statistics
        resource_summary = {}
        if self.resource_stats:
            cpu_values = [stat['cpu'] for stat in self.resource_stats]
            memory_values = [stat['memory'] for stat in self.resource_stats]
            
            resource_summary = {
                'avg_cpu': sum(cpu_values) / len(cpu_values),
                'max_cpu': max(cpu_values),
                'avg_memory': sum(memory_values) / len(memory_values),
                'max_memory': max(memory_values)
            }
        
        return {
            "query_metrics": {
                "total_queries": self.metrics.total_queries,
                "success_rate": self.metrics.success_rate,
                "avg_response_time": self.metrics.avg_response_time,
                "response_time_percentiles": percentiles
            },
            "cache_metrics": {
                "hit_rate": self.metrics.cache_hit_rate,
                "total_hits": self.metrics.cache_hits,
                "total_misses": self.metrics.cache_misses
            },
            "index_metrics": {
                "total_searches": self.metrics.index_searches,
                "avg_search_time": self.metrics.avg_index_search_time,
                "rebuilds": self.metrics.index_rebuilds
            },
            "resource_metrics": resource_summary,
            "alerts": self._check_performance_alerts()
        }
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        
        while not self.stop_monitoring.wait(30):  # Check every 30 seconds
            try:
                # Collect resource statistics
                self._collect_resource_stats()
                
                # Check for performance alerts
                alerts = self._check_performance_alerts()
                if alerts:
                    logger.warning(f"Performance alerts: {alerts}")
                    
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
    
    def _collect_resource_stats(self):
        """Collect system resource statistics"""
        
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            self.resource_stats.append({
                'timestamp': time.time(),
                'cpu': cpu_percent,
                'memory': memory.percent / 100.0,  # Convert to 0-1 scale
                'memory_available_gb': memory.available / (1024**3)
            })
            
            # Update peak metrics
            self.metrics.peak_memory_usage = max(
                self.metrics.peak_memory_usage, memory.percent / 100.0
            )
            self.metrics.avg_cpu_usage = cpu_percent  # Simplified for now
            
        except ImportError:
            # psutil not available
            pass
        except Exception as e:
            logger.warning(f"Resource collection error: {e}")
    
    def _check_performance_alerts(self) -> List[str]:
        """Check for performance issues and generate alerts"""
        
        alerts = []
        
        # Response time alert
        if self.metrics.avg_response_time > self.alert_thresholds['max_response_time']:
            alerts.append(f"High response time: {self.metrics.avg_response_time:.2f}s")
        
        # Cache hit rate alert
        if self.metrics.cache_hit_rate < self.alert_thresholds['min_cache_hit_rate']:
            alerts.append(f"Low cache hit rate: {self.metrics.cache_hit_rate:.2%}")
        
        # Memory usage alert
        if self.metrics.peak_memory_usage > self.alert_thresholds['max_memory_usage']:
            alerts.append(f"High memory usage: {self.metrics.peak_memory_usage:.2%}")
        
        return alerts

# Factory function for creating optimized components
async def create_optimized_rag_components(
    cache_config: CacheConfig = None,
    index_config: IndexConfig = None
) -> Tuple[AdvancedCache, OptimizedIndexManager, PerformanceMonitor]:
    """Create optimized RAG system components"""
    
    if cache_config is None:
        cache_config = CacheConfig()
    
    if index_config is None:
        index_config = IndexConfig()
    
    # Initialize components
    cache = AdvancedCache(cache_config)
    await cache.initialize()
    
    index_manager = OptimizedIndexManager(index_config)
    await index_manager.initialize()
    
    performance_monitor = PerformanceMonitor()
    performance_monitor.start_monitoring()
    
    logger.info("✅ Optimized RAG components created")
    
    return cache, index_manager, performance_monitor

# Example usage and testing
async def test_performance_optimization():
    """Test performance optimization components"""
    
    logger.info("🧪 Testing performance optimization components")
    
    # Create optimized components
    cache, index_manager, performance_monitor = await create_optimized_rag_components()
    
    try:
        # Test cache operations
        print("Testing cache operations...")
        
        # Test embedding caching
        test_embedding = np.random.random((384,)).astype(np.float32)
        await cache.set("test_embedding", test_embedding, category="embedding")
        
        cached_embedding = await cache.get("test_embedding", category="embedding")
        assert np.allclose(test_embedding, cached_embedding), "Embedding cache test failed"
        
        print("✅ Cache operations working")
        
        # Test index operations
        print("Testing index operations...")
        
        # Create test data
        n_vectors = 1000
        dimension = 384
        test_embeddings = np.random.random((n_vectors, dimension)).astype(np.float32)
        test_metadata = [{"id": i, "content": f"test content {i}"} for i in range(n_vectors)]
        
        # Build index
        success = await index_manager.build_optimized_index("test_content", test_embeddings, test_metadata)
        assert success, "Index building failed"
        
        # Test search
        query_embedding = np.random.random((dimension,)).astype(np.float32)
        results = await index_manager.optimized_search("test_content", query_embedding, k=5)
        
        assert len(results) == 5, f"Expected 5 results, got {len(results)}"
        print("✅ Index operations working")
        
        # Test performance monitoring
        print("Testing performance monitoring...")
        
        # Record some test metrics
        for i in range(10):
            performance_monitor.record_query(
                response_time=np.random.uniform(0.1, 2.0),
                success=np.random.choice([True, False], p=[0.9, 0.1]),
                cache_hit=np.random.choice([True, False], p=[0.6, 0.4])
            )
        
        # Get performance report
        report = performance_monitor.get_performance_report()
        print(f"Performance report: {json.dumps(report, indent=2)}")
        
        print("✅ Performance monitoring working")
        
        # Get component statistics
        cache_stats = cache.get_stats()
        index_stats = index_manager.get_stats()
        
        print(f"\nCache Statistics:")
        print(f"  Memory cache hit rate: {cache_stats['memory_cache']['hit_rate']:.2%}")
        print(f"  Redis cache hit rate: {cache_stats['redis_cache']['hit_rate']:.2%}")
        
        print(f"\nIndex Statistics:")
        print(f"  Total vectors: {index_stats['indices']['total_vectors']}")
        print(f"  Average search time: {index_stats['indices']['avg_search_time']:.4f}s")
        
        print("\n🎉 All performance optimization tests passed!")
        
    finally:
        # Cleanup
        await cache.close()
        await index_manager.close()
        performance_monitor.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(test_performance_optimization())