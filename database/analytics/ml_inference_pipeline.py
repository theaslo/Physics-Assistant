#!/usr/bin/env python3
"""
Real-Time ML Inference Pipeline with Model Monitoring for Physics Assistant Phase 6
Implements high-performance ML inference, model monitoring, A/B testing,
and real-time prediction serving with sub-100ms response times.
"""

import asyncio
import json
import logging
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import redis
import aioredis
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from enum import Enum
import uuid
import pickle
import warnings
from collections import defaultdict, deque
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
import uvloop
import msgpack
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Summary
import psutil

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    ADAPTIVE_LEARNING = "adaptive_learning"
    RISK_PREDICTION = "risk_prediction"
    RECOMMENDATION = "recommendation"
    NLP_ANALYSIS = "nlp_analysis"
    COMPUTER_VISION = "computer_vision"

class InferenceStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CACHE_HIT = "cache_hit"

@dataclass
class InferenceRequest:
    """ML inference request"""
    request_id: str
    model_type: ModelType
    model_version: str
    input_data: Dict[str, Any]
    student_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1=highest, 5=lowest
    timeout_ms: int = 100
    cache_enabled: bool = True
    ab_test_variant: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class InferenceResponse:
    """ML inference response"""
    request_id: str
    status: InferenceStatus
    prediction: Any
    confidence: float
    model_version: str
    processing_time_ms: float
    cache_hit: bool = False
    error_message: Optional[str] = None
    model_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    model_id: str
    model_version: str
    total_requests: int = 0
    successful_requests: int = 0
    error_requests: int = 0
    cache_hits: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    accuracy: float = 0.0
    prediction_drift: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class ABTestVariant:
    """A/B test variant configuration"""
    variant_id: str
    model_version: str
    traffic_percentage: float
    performance_metrics: ModelMetrics
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)

class ModelCache:
    """High-performance model result caching"""
    
    def __init__(self, redis_client: aioredis.Redis, default_ttl: int = 300):
        self.redis = redis_client
        self.default_ttl = default_ttl
        self.cache_hits = Counter('cache_hits_total', 'Total cache hits', ['model_type'])
        self.cache_misses = Counter('cache_misses_total', 'Total cache misses', ['model_type'])
    
    async def get(self, cache_key: str, model_type: str) -> Optional[Any]:
        """Get cached result"""
        try:
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                self.cache_hits.labels(model_type=model_type.value).inc()
                return msgpack.unpackb(cached_data, raw=False)
            else:
                self.cache_misses.labels(model_type=model_type.value).inc()
                return None
        except Exception as e:
            logger.error(f"❌ Cache get error: {e}")
            return None
    
    async def set(self, cache_key: str, result: Any, ttl: Optional[int] = None) -> bool:
        """Set cached result"""
        try:
            ttl = ttl or self.default_ttl
            packed_data = msgpack.packb(result, use_bin_type=True)
            await self.redis.setex(cache_key, ttl, packed_data)
            return True
        except Exception as e:
            logger.error(f"❌ Cache set error: {e}")
            return False
    
    def generate_cache_key(self, model_type: ModelType, model_version: str, 
                          input_data: Dict[str, Any]) -> str:
        """Generate cache key for request"""
        try:
            # Create deterministic hash of input data
            input_str = json.dumps(input_data, sort_keys=True)
            import hashlib
            input_hash = hashlib.md5(input_str.encode()).hexdigest()
            return f"ml_cache:{model_type.value}:{model_version}:{input_hash}"
        except Exception as e:
            logger.error(f"❌ Cache key generation error: {e}")
            return f"ml_cache:{model_type.value}:{model_version}:default"

class ModelMonitor:
    """Real-time model performance monitoring"""
    
    def __init__(self):
        # Prometheus metrics
        self.request_counter = Counter(
            'ml_requests_total', 
            'Total ML requests', 
            ['model_type', 'model_version', 'status']
        )
        
        self.latency_histogram = Histogram(
            'ml_inference_duration_seconds',
            'ML inference duration',
            ['model_type', 'model_version'],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        )
        
        self.confidence_histogram = Histogram(
            'ml_prediction_confidence',
            'ML prediction confidence scores',
            ['model_type', 'model_version'],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        
        self.error_counter = Counter(
            'ml_errors_total',
            'Total ML errors',
            ['model_type', 'model_version', 'error_type']
        )
        
        self.model_accuracy_gauge = Gauge(
            'ml_model_accuracy',
            'Current model accuracy',
            ['model_type', 'model_version']
        )
        
        self.prediction_drift_gauge = Gauge(
            'ml_prediction_drift',
            'Model prediction drift',
            ['model_type', 'model_version']
        )
        
        # Internal tracking
        self.model_metrics = defaultdict(lambda: ModelMetrics("", ""))
        self.latency_buffer = defaultdict(lambda: deque(maxlen=1000))
        self.prediction_buffer = defaultdict(lambda: deque(maxlen=1000))
        self.drift_baseline = defaultdict(dict)
    
    def record_request(self, request: InferenceRequest, response: InferenceResponse):
        """Record request metrics"""
        try:
            model_key = f"{request.model_type.value}:{request.model_version}"
            
            # Update Prometheus metrics
            self.request_counter.labels(
                model_type=request.model_type.value,
                model_version=request.model_version,
                status=response.status.value
            ).inc()
            
            self.latency_histogram.labels(
                model_type=request.model_type.value,
                model_version=request.model_version
            ).observe(response.processing_time_ms / 1000.0)
            
            if response.confidence is not None:
                self.confidence_histogram.labels(
                    model_type=request.model_type.value,
                    model_version=request.model_version
                ).observe(response.confidence)
            
            if response.status == InferenceStatus.ERROR:
                self.error_counter.labels(
                    model_type=request.model_type.value,
                    model_version=request.model_version,
                    error_type="inference_error"
                ).inc()
            
            # Update internal metrics
            metrics = self.model_metrics[model_key]
            metrics.model_id = model_key
            metrics.model_version = request.model_version
            metrics.total_requests += 1
            
            if response.status == InferenceStatus.SUCCESS:
                metrics.successful_requests += 1
            elif response.status == InferenceStatus.ERROR:
                metrics.error_requests += 1
            elif response.status == InferenceStatus.CACHE_HIT:
                metrics.cache_hits += 1
            
            # Update latency metrics
            self.latency_buffer[model_key].append(response.processing_time_ms)
            latencies = list(self.latency_buffer[model_key])
            if latencies:
                metrics.avg_latency_ms = np.mean(latencies)
                metrics.p95_latency_ms = np.percentile(latencies, 95)
                metrics.p99_latency_ms = np.percentile(latencies, 99)
            
            # Update prediction drift
            if response.prediction is not None:
                self.prediction_buffer[model_key].append(response.prediction)
                metrics.prediction_drift = self._calculate_prediction_drift(model_key)
                self.prediction_drift_gauge.labels(
                    model_type=request.model_type.value,
                    model_version=request.model_version
                ).set(metrics.prediction_drift)
            
            metrics.last_updated = datetime.now()
            
        except Exception as e:
            logger.error(f"❌ Failed to record request metrics: {e}")
    
    def _calculate_prediction_drift(self, model_key: str) -> float:
        """Calculate prediction drift from baseline"""
        try:
            predictions = list(self.prediction_buffer[model_key])
            if len(predictions) < 100:
                return 0.0
            
            # If no baseline, set current as baseline
            if model_key not in self.drift_baseline:
                self.drift_baseline[model_key] = {
                    'mean': np.mean(predictions[:50]),
                    'std': np.std(predictions[:50])
                }
                return 0.0
            
            # Calculate drift from baseline
            baseline = self.drift_baseline[model_key]
            recent_predictions = predictions[-50:]  # Last 50 predictions
            
            recent_mean = np.mean(recent_predictions)
            recent_std = np.std(recent_predictions)
            
            # Statistical distance from baseline
            mean_drift = abs(recent_mean - baseline['mean'])
            std_drift = abs(recent_std - baseline['std'])
            
            # Normalize drift (0-1 scale)
            drift_score = min(1.0, (mean_drift + std_drift) / 2.0)
            
            return drift_score
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate prediction drift: {e}")
            return 0.0
    
    def get_model_health(self, model_type: ModelType, model_version: str) -> Dict[str, Any]:
        """Get current model health status"""
        try:
            model_key = f"{model_type.value}:{model_version}"
            metrics = self.model_metrics[model_key]
            
            # Calculate health score
            health_score = 1.0
            
            # Error rate impact
            if metrics.total_requests > 0:
                error_rate = metrics.error_requests / metrics.total_requests
                health_score *= (1.0 - error_rate)
            
            # Latency impact
            if metrics.avg_latency_ms > 100:  # Target <100ms
                latency_penalty = min(0.5, (metrics.avg_latency_ms - 100) / 200)
                health_score *= (1.0 - latency_penalty)
            
            # Drift impact
            if metrics.prediction_drift > 0.5:
                drift_penalty = min(0.3, (metrics.prediction_drift - 0.5) / 0.5 * 0.3)
                health_score *= (1.0 - drift_penalty)
            
            health_status = "healthy"
            if health_score < 0.5:
                health_status = "critical"
            elif health_score < 0.7:
                health_status = "degraded"
            elif health_score < 0.9:
                health_status = "warning"
            
            return {
                'model_key': model_key,
                'health_score': health_score,
                'health_status': health_status,
                'metrics': {
                    'total_requests': metrics.total_requests,
                    'error_rate': metrics.error_requests / max(1, metrics.total_requests),
                    'avg_latency_ms': metrics.avg_latency_ms,
                    'p95_latency_ms': metrics.p95_latency_ms,
                    'prediction_drift': metrics.prediction_drift,
                    'cache_hit_rate': metrics.cache_hits / max(1, metrics.total_requests)
                },
                'last_updated': metrics.last_updated
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get model health: {e}")
            return {'health_status': 'unknown', 'health_score': 0.0}

class ABTestManager:
    """A/B testing for ML models"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.active_tests = {}
        self.variant_performance = defaultdict(lambda: ModelMetrics("", ""))
    
    async def create_ab_test(self, test_id: str, variants: List[ABTestVariant]) -> bool:
        """Create a new A/B test"""
        try:
            # Validate traffic percentages sum to 100%
            total_traffic = sum(variant.traffic_percentage for variant in variants)
            if abs(total_traffic - 100.0) > 0.01:
                raise ValueError(f"Traffic percentages must sum to 100%, got {total_traffic}")
            
            test_config = {
                'test_id': test_id,
                'variants': [
                    {
                        'variant_id': v.variant_id,
                        'model_version': v.model_version,
                        'traffic_percentage': v.traffic_percentage,
                        'is_active': v.is_active
                    }
                    for v in variants
                ],
                'created_at': datetime.now().isoformat(),
                'is_active': True
            }
            
            await self.redis.set(f"ab_test:{test_id}", json.dumps(test_config))
            self.active_tests[test_id] = test_config
            
            logger.info(f"✅ Created A/B test {test_id} with {len(variants)} variants")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create A/B test: {e}")
            return False
    
    async def get_variant_for_request(self, test_id: str, student_id: str) -> Optional[str]:
        """Get A/B test variant for a request"""
        try:
            if test_id not in self.active_tests:
                # Try to load from Redis
                test_data = await self.redis.get(f"ab_test:{test_id}")
                if test_data:
                    self.active_tests[test_id] = json.loads(test_data)
                else:
                    return None
            
            test_config = self.active_tests[test_id]
            if not test_config.get('is_active', False):
                return None
            
            # Deterministic assignment based on student_id
            import hashlib
            hash_input = f"{test_id}:{student_id}".encode()
            hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
            assignment_value = (hash_value % 10000) / 100.0  # 0-100 range
            
            # Find variant based on traffic percentages
            cumulative_percentage = 0.0
            for variant in test_config['variants']:
                cumulative_percentage += variant['traffic_percentage']
                if assignment_value <= cumulative_percentage:
                    return variant['variant_id']
            
            # Fallback to first variant
            return test_config['variants'][0]['variant_id'] if test_config['variants'] else None
            
        except Exception as e:
            logger.error(f"❌ Failed to get A/B test variant: {e}")
            return None
    
    def record_variant_performance(self, variant_id: str, request: InferenceRequest, 
                                 response: InferenceResponse):
        """Record performance metrics for A/B test variant"""
        try:
            metrics = self.variant_performance[variant_id]
            metrics.model_id = variant_id
            metrics.model_version = request.model_version
            metrics.total_requests += 1
            
            if response.status == InferenceStatus.SUCCESS:
                metrics.successful_requests += 1
                
            # Update latency metrics
            latency_key = f"{variant_id}_latency"
            if not hasattr(self, '_variant_latencies'):
                self._variant_latencies = defaultdict(lambda: deque(maxlen=1000))
            
            self._variant_latencies[latency_key].append(response.processing_time_ms)
            latencies = list(self._variant_latencies[latency_key])
            if latencies:
                metrics.avg_latency_ms = np.mean(latencies)
                metrics.p95_latency_ms = np.percentile(latencies, 95)
            
            metrics.last_updated = datetime.now()
            
        except Exception as e:
            logger.error(f"❌ Failed to record variant performance: {e}")
    
    async def get_ab_test_results(self, test_id: str) -> Dict[str, Any]:
        """Get A/B test performance results"""
        try:
            if test_id not in self.active_tests:
                return {}
            
            test_config = self.active_tests[test_id]
            results = {
                'test_id': test_id,
                'variants': [],
                'statistical_significance': None
            }
            
            for variant in test_config['variants']:
                variant_id = variant['variant_id']
                metrics = self.variant_performance[variant_id]
                
                conversion_rate = metrics.successful_requests / max(1, metrics.total_requests)
                
                results['variants'].append({
                    'variant_id': variant_id,
                    'model_version': variant['model_version'],
                    'traffic_percentage': variant['traffic_percentage'],
                    'total_requests': metrics.total_requests,
                    'conversion_rate': conversion_rate,
                    'avg_latency_ms': metrics.avg_latency_ms,
                    'performance_score': conversion_rate * (1.0 - metrics.avg_latency_ms / 1000.0)
                })
            
            # Calculate statistical significance (simplified)
            if len(results['variants']) == 2:
                v1, v2 = results['variants']
                if v1['total_requests'] > 100 and v2['total_requests'] > 100:
                    # Simplified significance test
                    rate_diff = abs(v1['conversion_rate'] - v2['conversion_rate'])
                    results['statistical_significance'] = rate_diff > 0.05
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to get A/B test results: {e}")
            return {}

class ModelLoader:
    """Dynamic model loading and management"""
    
    def __init__(self):
        self.loaded_models = {}
        self.model_configs = {}
        self.load_lock = threading.Lock()
    
    async def load_model(self, model_type: ModelType, model_version: str, 
                        model_path: str) -> bool:
        """Load ML model into memory"""
        try:
            model_key = f"{model_type.value}:{model_version}"
            
            with self.load_lock:
                if model_key in self.loaded_models:
                    return True
                
                # Load model based on type
                if model_type == ModelType.ADAPTIVE_LEARNING:
                    model = await self._load_pytorch_model(model_path)
                elif model_type == ModelType.RISK_PREDICTION:
                    model = await self._load_sklearn_model(model_path)
                elif model_type == ModelType.RECOMMENDATION:
                    model = await self._load_sklearn_model(model_path)
                else:
                    model = await self._load_generic_model(model_path)
                
                if model:
                    self.loaded_models[model_key] = model
                    self.model_configs[model_key] = {
                        'model_type': model_type,
                        'model_version': model_version,
                        'model_path': model_path,
                        'loaded_at': datetime.now(),
                        'last_used': datetime.now()
                    }
                    logger.info(f"✅ Loaded model {model_key}")
                    return True
                
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to load model {model_type.value}:{model_version}: {e}")
            return False
    
    async def _load_pytorch_model(self, model_path: str):
        """Load PyTorch model"""
        try:
            model = torch.load(model_path, map_location='cpu')
            model.eval()
            return model
        except Exception as e:
            logger.error(f"❌ Failed to load PyTorch model: {e}")
            return None
    
    async def _load_sklearn_model(self, model_path: str):
        """Load scikit-learn model"""
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            logger.error(f"❌ Failed to load sklearn model: {e}")
            return None
    
    async def _load_generic_model(self, model_path: str):
        """Load generic model"""
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            logger.error(f"❌ Failed to load generic model: {e}")
            return None
    
    def get_model(self, model_type: ModelType, model_version: str):
        """Get loaded model"""
        model_key = f"{model_type.value}:{model_version}"
        if model_key in self.loaded_models:
            # Update last used time
            if model_key in self.model_configs:
                self.model_configs[model_key]['last_used'] = datetime.now()
            return self.loaded_models[model_key]
        return None
    
    async def unload_model(self, model_type: ModelType, model_version: str) -> bool:
        """Unload model from memory"""
        try:
            model_key = f"{model_type.value}:{model_version}"
            
            with self.load_lock:
                if model_key in self.loaded_models:
                    del self.loaded_models[model_key]
                if model_key in self.model_configs:
                    del self.model_configs[model_key]
                
                logger.info(f"✅ Unloaded model {model_key}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to unload model: {e}")
            return False

class MLInferencePipeline:
    """High-performance ML inference pipeline"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", max_workers: int = None):
        self.redis_url = redis_url
        self.redis_client = None
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) + 4)
        
        # Components
        self.model_loader = ModelLoader()
        self.model_cache = None
        self.model_monitor = ModelMonitor()
        self.ab_test_manager = None
        
        # Request queue and processing
        self.request_queue = asyncio.Queue(maxsize=10000)
        self.worker_pool = None
        self.processing_active = False
        
        # Performance metrics
        self.queue_depth_gauge = Gauge('ml_queue_depth', 'Current inference queue depth')
        self.worker_utilization_gauge = Gauge('ml_worker_utilization', 'Worker utilization percentage')
        
        # Rate limiting
        self.rate_limiter = defaultdict(lambda: deque(maxlen=100))
        self.rate_limit_per_minute = 1000
    
    async def initialize(self):
        """Initialize the ML inference pipeline"""
        try:
            logger.info("🚀 Initializing ML Inference Pipeline")
            
            # Initialize Redis connection
            self.redis_client = await aioredis.from_url(self.redis_url, decode_responses=False)
            
            # Initialize components
            self.model_cache = ModelCache(self.redis_client)
            self.ab_test_manager = ABTestManager(self.redis_client)
            
            # Initialize worker pool
            self.worker_pool = ThreadPoolExecutor(max_workers=self.max_workers)
            
            # Start processing workers
            self.processing_active = True
            asyncio.create_task(self._start_processing_workers())
            
            # Start monitoring tasks
            asyncio.create_task(self._start_monitoring_tasks())
            
            logger.info(f"✅ ML Inference Pipeline initialized with {self.max_workers} workers")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ML Inference Pipeline: {e}")
            return False
    
    async def predict(self, request: InferenceRequest) -> InferenceResponse:
        """Make ML prediction"""
        start_time = time.time()
        
        try:
            # Rate limiting check
            if not self._check_rate_limit(request.student_id):
                return InferenceResponse(
                    request_id=request.request_id,
                    status=InferenceStatus.ERROR,
                    prediction=None,
                    confidence=0.0,
                    model_version=request.model_version,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    error_message="Rate limit exceeded"
                )
            
            # Check cache first
            if request.cache_enabled:
                cache_key = self.model_cache.generate_cache_key(
                    request.model_type, request.model_version, request.input_data
                )
                cached_result = await self.model_cache.get(cache_key, request.model_type)
                
                if cached_result:
                    response = InferenceResponse(
                        request_id=request.request_id,
                        status=InferenceStatus.CACHE_HIT,
                        prediction=cached_result['prediction'],
                        confidence=cached_result['confidence'],
                        model_version=request.model_version,
                        processing_time_ms=(time.time() - start_time) * 1000,
                        cache_hit=True
                    )
                    self.model_monitor.record_request(request, response)
                    return response
            
            # A/B testing
            if request.ab_test_variant:
                variant_model_version = await self._get_variant_model_version(
                    request.ab_test_variant
                )
                if variant_model_version:
                    request.model_version = variant_model_version
            
            # Queue request for processing
            try:
                await asyncio.wait_for(
                    self.request_queue.put(request),
                    timeout=request.timeout_ms / 1000.0
                )
                
                # Process request
                response = await self._process_inference_request(request, start_time)
                
                # Cache result if successful
                if (response.status == InferenceStatus.SUCCESS and 
                    request.cache_enabled and response.prediction is not None):
                    cache_key = self.model_cache.generate_cache_key(
                        request.model_type, request.model_version, request.input_data
                    )
                    await self.model_cache.set(cache_key, {
                        'prediction': response.prediction,
                        'confidence': response.confidence
                    })
                
                # Record metrics
                self.model_monitor.record_request(request, response)
                
                # Record A/B test metrics
                if request.ab_test_variant:
                    self.ab_test_manager.record_variant_performance(
                        request.ab_test_variant, request, response
                    )
                
                return response
                
            except asyncio.TimeoutError:
                return InferenceResponse(
                    request_id=request.request_id,
                    status=InferenceStatus.TIMEOUT,
                    prediction=None,
                    confidence=0.0,
                    model_version=request.model_version,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    error_message="Request timeout"
                )
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            response = InferenceResponse(
                request_id=request.request_id,
                status=InferenceStatus.ERROR,
                prediction=None,
                confidence=0.0,
                model_version=request.model_version,
                processing_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
            self.model_monitor.record_request(request, response)
            return response
    
    def _check_rate_limit(self, student_id: Optional[str]) -> bool:
        """Check if request is within rate limits"""
        try:
            if not student_id:
                return True
            
            current_time = time.time()
            requests = self.rate_limiter[student_id]
            
            # Remove old requests (older than 1 minute)
            while requests and current_time - requests[0] > 60:
                requests.popleft()
            
            # Check rate limit
            if len(requests) >= self.rate_limit_per_minute:
                return False
            
            # Add current request
            requests.append(current_time)
            return True
            
        except Exception as e:
            logger.error(f"❌ Rate limit check error: {e}")
            return True  # Allow request on error
    
    async def _process_inference_request(self, request: InferenceRequest, 
                                       start_time: float) -> InferenceResponse:
        """Process individual inference request"""
        try:
            # Get model
            model = self.model_loader.get_model(request.model_type, request.model_version)
            
            if not model:
                return InferenceResponse(
                    request_id=request.request_id,
                    status=InferenceStatus.ERROR,
                    prediction=None,
                    confidence=0.0,
                    model_version=request.model_version,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    error_message="Model not loaded"
                )
            
            # Process based on model type
            if request.model_type == ModelType.ADAPTIVE_LEARNING:
                prediction, confidence = await self._process_adaptive_learning(model, request.input_data)
            elif request.model_type == ModelType.RISK_PREDICTION:
                prediction, confidence = await self._process_risk_prediction(model, request.input_data)
            elif request.model_type == ModelType.RECOMMENDATION:
                prediction, confidence = await self._process_recommendation(model, request.input_data)
            else:
                prediction, confidence = await self._process_generic_model(model, request.input_data)
            
            return InferenceResponse(
                request_id=request.request_id,
                status=InferenceStatus.SUCCESS,
                prediction=prediction,
                confidence=confidence,
                model_version=request.model_version,
                processing_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            logger.error(f"❌ Inference processing error: {e}")
            return InferenceResponse(
                request_id=request.request_id,
                status=InferenceStatus.ERROR,
                prediction=None,
                confidence=0.0,
                model_version=request.model_version,
                processing_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
    
    async def _process_adaptive_learning(self, model, input_data: Dict[str, Any]) -> Tuple[Any, float]:
        """Process adaptive learning model"""
        try:
            # Extract features
            features = np.array(input_data.get('features', []))
            if len(features) == 0:
                return None, 0.0
            
            # Convert to torch tensor
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(features_tensor)
                
                if isinstance(outputs, dict):
                    prediction = outputs.get('overall_risk', torch.tensor([0.5]))
                    uncertainty = outputs.get('uncertainties', torch.tensor([0.1]))
                else:
                    prediction = outputs
                    uncertainty = torch.tensor([0.1])
                
                pred_value = float(prediction.squeeze())
                confidence = 1.0 - float(uncertainty.squeeze())
                
                return pred_value, confidence
            
        except Exception as e:
            logger.error(f"❌ Adaptive learning processing error: {e}")
            return None, 0.0
    
    async def _process_risk_prediction(self, model, input_data: Dict[str, Any]) -> Tuple[Any, float]:
        """Process risk prediction model"""
        try:
            # Extract features
            features = np.array(input_data.get('features', []))
            if len(features) == 0:
                return None, 0.0
            
            # Make prediction
            prediction = model.predict_proba([features])
            risk_probability = float(prediction[0][1])  # Probability of high risk
            
            # Get confidence from model if available
            if hasattr(model, 'predict_confidence'):
                confidence = model.predict_confidence([features])[0]
            else:
                # Use prediction certainty as confidence proxy
                confidence = max(prediction[0]) * 2 - 1  # Convert to 0-1 scale
            
            return risk_probability, confidence
            
        except Exception as e:
            logger.error(f"❌ Risk prediction processing error: {e}")
            return None, 0.0
    
    async def _process_recommendation(self, model, input_data: Dict[str, Any]) -> Tuple[Any, float]:
        """Process recommendation model"""
        try:
            # Extract user and item features
            user_features = np.array(input_data.get('user_features', []))
            item_features = np.array(input_data.get('item_features', []))
            
            if len(user_features) == 0 or len(item_features) == 0:
                return [], 0.0
            
            # Make recommendation
            recommendations = model.predict([user_features])
            confidence = 0.8  # Default confidence for recommendations
            
            return recommendations.tolist(), confidence
            
        except Exception as e:
            logger.error(f"❌ Recommendation processing error: {e}")
            return [], 0.0
    
    async def _process_generic_model(self, model, input_data: Dict[str, Any]) -> Tuple[Any, float]:
        """Process generic model"""
        try:
            features = np.array(input_data.get('features', []))
            if len(features) == 0:
                return None, 0.0
            
            prediction = model.predict([features])[0]
            confidence = 0.7  # Default confidence
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"❌ Generic model processing error: {e}")
            return None, 0.0
    
    async def _get_variant_model_version(self, variant_id: str) -> Optional[str]:
        """Get model version for A/B test variant"""
        try:
            # This would lookup the variant configuration
            # For now, return the variant_id as model version
            return variant_id
            
        except Exception as e:
            logger.error(f"❌ Failed to get variant model version: {e}")
            return None
    
    async def _start_processing_workers(self):
        """Start background processing workers"""
        try:
            workers = []
            for i in range(min(4, self.max_workers)):
                worker = asyncio.create_task(self._processing_worker(f"worker_{i}"))
                workers.append(worker)
            
            await asyncio.gather(*workers)
            
        except Exception as e:
            logger.error(f"❌ Processing workers error: {e}")
    
    async def _processing_worker(self, worker_id: str):
        """Individual processing worker"""
        try:
            while self.processing_active:
                try:
                    # Update queue depth metric
                    self.queue_depth_gauge.set(self.request_queue.qsize())
                    
                    # Get next request with timeout
                    request = await asyncio.wait_for(
                        self.request_queue.get(), timeout=1.0
                    )
                    
                    # Mark task as done
                    self.request_queue.task_done()
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"❌ Worker {worker_id} error: {e}")
                    continue
                
        except Exception as e:
            logger.error(f"❌ Worker {worker_id} fatal error: {e}")
    
    async def _start_monitoring_tasks(self):
        """Start background monitoring tasks"""
        try:
            # System monitoring
            asyncio.create_task(self._monitor_system_resources())
            
            # Model health monitoring
            asyncio.create_task(self._monitor_model_health())
            
        except Exception as e:
            logger.error(f"❌ Monitoring tasks error: {e}")
    
    async def _monitor_system_resources(self):
        """Monitor system resource usage"""
        try:
            cpu_gauge = Gauge('system_cpu_usage', 'System CPU usage percentage')
            memory_gauge = Gauge('system_memory_usage', 'System memory usage percentage')
            
            while self.processing_active:
                try:
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    cpu_gauge.set(cpu_percent)
                    
                    # Memory usage
                    memory = psutil.virtual_memory()
                    memory_gauge.set(memory.percent)
                    
                    # Worker utilization
                    active_workers = self.max_workers - self.request_queue.qsize()
                    utilization = (active_workers / self.max_workers) * 100
                    self.worker_utilization_gauge.set(utilization)
                    
                    await asyncio.sleep(30)  # Monitor every 30 seconds
                    
                except Exception as e:
                    logger.error(f"❌ System monitoring error: {e}")
                    await asyncio.sleep(30)
                    
        except Exception as e:
            logger.error(f"❌ System monitoring fatal error: {e}")
    
    async def _monitor_model_health(self):
        """Monitor model health and trigger alerts"""
        try:
            while self.processing_active:
                try:
                    # Check health of all loaded models
                    for model_key, config in self.model_loader.model_configs.items():
                        model_type = config['model_type']
                        model_version = config['model_version']
                        
                        health = self.model_monitor.get_model_health(model_type, model_version)
                        
                        # Alert on unhealthy models
                        if health['health_status'] in ['critical', 'degraded']:
                            logger.warning(f"⚠️ Model {model_key} health: {health['health_status']} (score: {health['health_score']:.2f})")
                        
                        # Auto-unload unused models
                        last_used = config.get('last_used', datetime.now())
                        if (datetime.now() - last_used).total_seconds() > 3600:  # 1 hour
                            await self.model_loader.unload_model(model_type, model_version)
                    
                    await asyncio.sleep(60)  # Check every minute
                    
                except Exception as e:
                    logger.error(f"❌ Model health monitoring error: {e}")
                    await asyncio.sleep(60)
                    
        except Exception as e:
            logger.error(f"❌ Model health monitoring fatal error: {e}")
    
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        try:
            return {
                'processing_active': self.processing_active,
                'queue_depth': self.request_queue.qsize(),
                'max_workers': self.max_workers,
                'loaded_models': len(self.model_loader.loaded_models),
                'model_configs': list(self.model_loader.model_configs.keys()),
                'system_resources': {
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'available_memory_gb': psutil.virtual_memory().available / (1024**3)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get pipeline status: {e}")
            return {'error': str(e)}
    
    async def shutdown(self):
        """Gracefully shutdown the pipeline"""
        try:
            logger.info("🛑 Shutting down ML Inference Pipeline")
            
            self.processing_active = False
            
            # Wait for queue to empty
            await self.request_queue.join()
            
            # Shutdown worker pool
            if self.worker_pool:
                self.worker_pool.shutdown(wait=True)
            
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("✅ ML Inference Pipeline shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Pipeline shutdown error: {e}")

# Testing function
async def test_ml_inference_pipeline():
    """Test the ML inference pipeline"""
    try:
        logger.info("🧪 Testing ML Inference Pipeline")
        
        pipeline = MLInferencePipeline()
        await pipeline.initialize()
        
        # Test inference request
        request = InferenceRequest(
            request_id=str(uuid.uuid4()),
            model_type=ModelType.RISK_PREDICTION,
            model_version="v1.0",
            input_data={'features': [0.1, 0.2, 0.3, 0.4, 0.5]},
            student_id="test_student",
            timeout_ms=100
        )
        
        # Note: This would require actual models to be loaded
        # response = await pipeline.predict(request)
        # logger.info(f"✅ Prediction completed: {response.status.value}")
        
        # Test pipeline status
        status = await pipeline.get_pipeline_status()
        logger.info(f"✅ Pipeline status: {status['processing_active']}")
        
        await pipeline.shutdown()
        logger.info("✅ ML Inference Pipeline test completed")
        
    except Exception as e:
        logger.error(f"❌ ML Inference Pipeline test failed: {e}")

if __name__ == "__main__":
    # Use uvloop for better performance
    uvloop.install()
    asyncio.run(test_ml_inference_pipeline())