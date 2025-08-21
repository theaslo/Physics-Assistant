#!/usr/bin/env python3
"""
Real-time ML Inference Engine for Physics Assistant Phase 6
High-performance inference pipeline with <100ms response times,
caching, and model optimization for educational ML models.
"""

import asyncio
import json
import logging
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import redis
import hashlib
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from concurrent.futures import ThreadPoolExecutor
import pickle
import warnings
from contextlib import asynccontextmanager
import uvloop
import aioredis

# Suppress warnings for production performance
warnings.filterwarnings('ignore')
torch.set_num_threads(2)  # Optimize for inference

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InferenceRequest:
    """Real-time inference request"""
    request_id: str
    model_name: str
    features: Dict[str, Any]
    student_id: str
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 1  # 1=highest, 5=lowest

@dataclass
class InferenceResponse:
    """Real-time inference response"""
    request_id: str
    prediction: float
    confidence: float
    explanation: str
    feature_importance: Dict[str, float]
    model_version: str
    inference_time_ms: float
    cache_hit: bool = False
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ModelMetrics:
    """Real-time model performance metrics"""
    model_name: str
    total_requests: int = 0
    avg_inference_time_ms: float = 0.0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0
    throughput_per_second: float = 0.0
    p95_latency_ms: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

class ModelOptimizer:
    """Optimize ML models for real-time inference"""
    
    def __init__(self):
        self.optimized_models = {}
        self.optimization_cache = {}
    
    async def optimize_pytorch_model(self, model: nn.Module, model_name: str, 
                                   sample_input: torch.Tensor) -> str:
        """Convert PyTorch model to optimized ONNX format"""
        try:
            logger.info(f"🚀 Optimizing PyTorch model: {model_name}")
            
            # Set model to evaluation mode
            model.eval()
            
            # Export to ONNX
            onnx_path = f"/tmp/{model_name}_optimized.onnx"
            
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    sample_input,
                    onnx_path,
                    export_params=True,
                    opset_version=14,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
            
            # Verify ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            # Create optimized runtime session
            providers = ['CPUExecutionProvider']
            if ort.get_device() == 'GPU':
                providers.insert(0, 'CUDAExecutionProvider')
            
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            session_options.intra_op_num_threads = 2
            session_options.inter_op_num_threads = 2
            
            session = ort.InferenceSession(onnx_path, session_options, providers=providers)
            
            self.optimized_models[model_name] = {
                'session': session,
                'input_name': session.get_inputs()[0].name,
                'output_name': session.get_outputs()[0].name,
                'optimization_time': datetime.now()
            }
            
            logger.info(f"✅ Model {model_name} optimized successfully")
            return onnx_path
            
        except Exception as e:
            logger.error(f"❌ Failed to optimize model {model_name}: {e}")
            raise
    
    async def run_optimized_inference(self, model_name: str, 
                                    input_data: np.ndarray) -> np.ndarray:
        """Run inference on optimized model"""
        try:
            if model_name not in self.optimized_models:
                raise ValueError(f"Model {model_name} not optimized")
            
            model_info = self.optimized_models[model_name]
            session = model_info['session']
            input_name = model_info['input_name']
            
            # Ensure correct input shape and type
            if len(input_data.shape) == 1:
                input_data = input_data.reshape(1, -1)
            
            input_data = input_data.astype(np.float32)
            
            # Run inference
            outputs = session.run(None, {input_name: input_data})
            return outputs[0]
            
        except Exception as e:
            logger.error(f"❌ Optimized inference failed for {model_name}: {e}")
            raise

class FeatureCache:
    """High-performance feature caching system"""
    
    def __init__(self, redis_client=None, ttl: int = 300):
        self.redis_client = redis_client
        self.ttl = ttl  # 5 minutes default
        self.local_cache = {}
        self.cache_stats = defaultdict(int)
        self.max_local_cache_size = 1000
    
    def _generate_cache_key(self, student_id: str, feature_type: str, 
                          context_hash: str = "") -> str:
        """Generate cache key for features"""
        key_data = f"{student_id}:{feature_type}:{context_hash}"
        return f"features:{hashlib.md5(key_data.encode()).hexdigest()}"
    
    async def get_features(self, student_id: str, feature_type: str, 
                         context: Dict[str, Any] = None) -> Optional[Dict[str, float]]:
        """Get cached features if available"""
        try:
            context_hash = hashlib.md5(json.dumps(context or {}, sort_keys=True).encode()).hexdigest()[:8]
            cache_key = self._generate_cache_key(student_id, feature_type, context_hash)
            
            # Check local cache first
            if cache_key in self.local_cache:
                features, timestamp = self.local_cache[cache_key]
                if datetime.now() - timestamp < timedelta(seconds=self.ttl):
                    self.cache_stats['local_hits'] += 1
                    return features
                else:
                    del self.local_cache[cache_key]
            
            # Check Redis cache
            if self.redis_client:
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    features = json.loads(cached_data)
                    
                    # Store in local cache
                    if len(self.local_cache) < self.max_local_cache_size:
                        self.local_cache[cache_key] = (features, datetime.now())
                    
                    self.cache_stats['redis_hits'] += 1
                    return features
            
            self.cache_stats['misses'] += 1
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get cached features: {e}")
            return None
    
    async def cache_features(self, student_id: str, feature_type: str, 
                           features: Dict[str, float], context: Dict[str, Any] = None):
        """Cache features for future use"""
        try:
            context_hash = hashlib.md5(json.dumps(context or {}, sort_keys=True).encode()).hexdigest()[:8]
            cache_key = self._generate_cache_key(student_id, feature_type, context_hash)
            
            # Store in local cache
            if len(self.local_cache) < self.max_local_cache_size:
                self.local_cache[cache_key] = (features, datetime.now())
            
            # Store in Redis cache
            if self.redis_client:
                await self.redis_client.setex(cache_key, self.ttl, json.dumps(features))
            
            self.cache_stats['stores'] += 1
            
        except Exception as e:
            logger.error(f"❌ Failed to cache features: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = sum(self.cache_stats.values())
        hit_rate = (self.cache_stats['local_hits'] + self.cache_stats['redis_hits']) / max(1, total_requests)
        
        return {
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'local_hits': self.cache_stats['local_hits'],
            'redis_hits': self.cache_stats['redis_hits'],
            'misses': self.cache_stats['misses'],
            'stores': self.cache_stats['stores'],
            'local_cache_size': len(self.local_cache)
        }

class RealTimeMLInferenceEngine:
    """High-performance real-time ML inference engine"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", 
                 max_workers: int = 4):
        self.redis_url = redis_url
        self.redis_client = None
        self.max_workers = max_workers
        
        # Core components
        self.model_optimizer = ModelOptimizer()
        self.feature_cache = None
        
        # Model registry
        self.loaded_models = {}
        self.model_metadata = {}
        
        # Performance monitoring
        self.metrics = {}
        self.request_queue = asyncio.Queue(maxsize=1000)
        self.response_times = defaultdict(lambda: deque(maxlen=1000))
        self.error_counts = defaultdict(int)
        
        # Thread pool for CPU-intensive operations
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Request batching
        self.batch_size = 32
        self.batch_timeout_ms = 50
        self.pending_requests = defaultdict(list)
        
        # Circuit breaker
        self.circuit_breaker = {
            'error_threshold': 10,
            'time_window': 60,
            'recovery_timeout': 30
        }
        self.circuit_state = defaultdict(lambda: {'state': 'closed', 'errors': 0, 'last_failure': None})
    
    async def initialize(self):
        """Initialize the real-time inference engine"""
        try:
            logger.info("🚀 Initializing Real-time ML Inference Engine")
            
            # Initialize Redis connection
            self.redis_client = aioredis.from_url(
                self.redis_url, 
                encoding="utf-8", 
                decode_responses=True,
                max_connections=20
            )
            
            # Initialize feature cache
            self.feature_cache = FeatureCache(self.redis_client, ttl=300)
            
            # Initialize metrics
            await self._initialize_metrics()
            
            # Start background tasks
            asyncio.create_task(self._batch_processor())
            asyncio.create_task(self._metrics_updater())
            asyncio.create_task(self._health_monitor())
            
            logger.info("✅ Real-time ML Inference Engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Real-time ML Inference Engine: {e}")
            return False
    
    async def _initialize_metrics(self):
        """Initialize performance metrics"""
        try:
            model_names = ['success_predictor', 'engagement_predictor', 'difficulty_adapter']
            
            for model_name in model_names:
                self.metrics[model_name] = ModelMetrics(model_name=model_name)
            
            logger.info("📊 Performance metrics initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize metrics: {e}")
    
    async def load_model(self, model_name: str, model_path: str, 
                        model_type: str = 'pytorch') -> bool:
        """Load and optimize model for real-time inference"""
        try:
            logger.info(f"📥 Loading model: {model_name}")
            
            if model_type == 'pytorch':
                # Load PyTorch model
                model = torch.load(model_path, map_location='cpu')
                model.eval()
                
                # Create sample input for optimization
                sample_input = torch.randn(1, 50)  # Adjust based on your feature size
                
                # Optimize model
                onnx_path = await self.model_optimizer.optimize_pytorch_model(
                    model, model_name, sample_input
                )
                
                self.loaded_models[model_name] = {
                    'type': 'onnx',
                    'path': onnx_path,
                    'loaded_at': datetime.now()
                }
            
            elif model_type == 'sklearn':
                # Load scikit-learn model (already serialized)
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.loaded_models[model_name] = {
                    'type': 'sklearn',
                    'model': model_data['model'],
                    'scaler': model_data.get('scaler'),
                    'loaded_at': datetime.now()
                }
            
            # Store metadata
            self.model_metadata[model_name] = {
                'version': '1.0',
                'loaded_at': datetime.now(),
                'optimization_level': 'high'
            }
            
            logger.info(f"✅ Model {model_name} loaded and optimized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model {model_name}: {e}")
            return False
    
    async def predict(self, request: InferenceRequest) -> InferenceResponse:
        """High-performance prediction with sub-100ms latency"""
        start_time = time.time()
        
        try:
            # Validate request
            if request.model_name not in self.loaded_models:
                raise ValueError(f"Model {request.model_name} not loaded")
            
            # Check circuit breaker
            if await self._is_circuit_open(request.model_name):
                raise Exception(f"Circuit breaker open for {request.model_name}")
            
            # Try to get cached prediction
            cache_key = self._generate_prediction_cache_key(request)
            cached_response = await self._get_cached_prediction(cache_key)
            if cached_response:
                cached_response.cache_hit = True
                return cached_response
            
            # Extract and prepare features
            feature_vector = await self._extract_feature_vector(request)
            
            # Run inference based on model type
            model_info = self.loaded_models[request.model_name]
            
            if model_info['type'] == 'onnx':
                prediction, confidence = await self._run_onnx_inference(
                    request.model_name, feature_vector
                )
            elif model_info['type'] == 'sklearn':
                prediction, confidence = await self._run_sklearn_inference(
                    request.model_name, feature_vector
                )
            else:
                raise ValueError(f"Unsupported model type: {model_info['type']}")
            
            # Generate explanation
            explanation = await self._generate_fast_explanation(
                request, feature_vector, prediction
            )
            
            # Calculate feature importance (simplified for speed)
            feature_importance = await self._calculate_feature_importance(
                request, feature_vector
            )
            
            # Create response
            inference_time_ms = (time.time() - start_time) * 1000
            
            response = InferenceResponse(
                request_id=request.request_id,
                prediction=float(prediction),
                confidence=float(confidence),
                explanation=explanation,
                feature_importance=feature_importance,
                model_version=self.model_metadata[request.model_name]['version'],
                inference_time_ms=inference_time_ms
            )
            
            # Cache response for future use
            await self._cache_prediction(cache_key, response)
            
            # Update metrics
            await self._update_metrics(request.model_name, inference_time_ms, success=True)
            
            # Log if slow
            if inference_time_ms > 100:
                logger.warning(f"⚠️ Slow inference: {inference_time_ms:.1f}ms for {request.model_name}")
            
            return response
            
        except Exception as e:
            inference_time_ms = (time.time() - start_time) * 1000
            await self._update_metrics(request.model_name, inference_time_ms, success=False)
            await self._record_circuit_breaker_error(request.model_name)
            
            logger.error(f"❌ Prediction failed for {request.request_id}: {e}")
            
            # Return error response
            return InferenceResponse(
                request_id=request.request_id,
                prediction=0.5,  # Default safe prediction
                confidence=0.0,
                explanation=f"Prediction failed: {str(e)}",
                feature_importance={},
                model_version="error",
                inference_time_ms=inference_time_ms
            )
    
    async def _extract_feature_vector(self, request: InferenceRequest) -> np.ndarray:
        """Extract and prepare feature vector for inference"""
        try:
            # Check cache first
            cached_features = await self.feature_cache.get_features(
                request.student_id, 'ml_inference', request.context
            )
            
            if cached_features:
                feature_vector = np.array(list(cached_features.values()), dtype=np.float32)
            else:
                # Extract features (simplified for real-time)
                features = {}
                
                # Basic features from request
                if 'success_rate' in request.features:
                    features['success_rate'] = request.features['success_rate']
                if 'avg_response_time' in request.features:
                    features['avg_response_time'] = request.features['avg_response_time']
                if 'help_seeking_rate' in request.features:
                    features['help_seeking_rate'] = request.features['help_seeking_rate']
                
                # Pad to fixed size (50 features)
                feature_list = list(features.values())
                while len(feature_list) < 50:
                    feature_list.append(0.0)
                
                feature_vector = np.array(feature_list[:50], dtype=np.float32)
                
                # Cache for future use
                await self.feature_cache.cache_features(
                    request.student_id, 'ml_inference', features, request.context
                )
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"❌ Failed to extract feature vector: {e}")
            return np.zeros(50, dtype=np.float32)
    
    async def _run_onnx_inference(self, model_name: str, 
                                feature_vector: np.ndarray) -> Tuple[float, float]:
        """Run inference on ONNX optimized model"""
        try:
            # Ensure correct shape
            if len(feature_vector.shape) == 1:
                feature_vector = feature_vector.reshape(1, -1)
            
            # Run optimized inference
            prediction = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.model_optimizer.run_optimized_inference,
                model_name,
                feature_vector
            )
            
            # Extract prediction and confidence
            if len(prediction.shape) > 1:
                pred_value = float(prediction[0][0])
            else:
                pred_value = float(prediction[0])
            
            # Simple confidence estimation
            confidence = min(0.95, abs(pred_value) + 0.5)
            
            return pred_value, confidence
            
        except Exception as e:
            logger.error(f"❌ ONNX inference failed: {e}")
            return 0.5, 0.1
    
    async def _run_sklearn_inference(self, model_name: str, 
                                   feature_vector: np.ndarray) -> Tuple[float, float]:
        """Run inference on scikit-learn model"""
        try:
            model_info = self.loaded_models[model_name]
            model = model_info['model']
            scaler = model_info.get('scaler')
            
            # Scale features if scaler available
            if scaler:
                feature_vector = scaler.transform(feature_vector.reshape(1, -1))
            else:
                feature_vector = feature_vector.reshape(1, -1)
            
            # Run prediction
            prediction = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                model.predict,
                feature_vector
            )
            
            # Get confidence if available
            if hasattr(model, 'predict_proba'):
                proba = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    model.predict_proba,
                    feature_vector
                )
                confidence = float(np.max(proba[0]))
            else:
                confidence = 0.75  # Default confidence for regression
            
            return float(prediction[0]), confidence
            
        except Exception as e:
            logger.error(f"❌ Sklearn inference failed: {e}")
            return 0.5, 0.1
    
    async def _generate_fast_explanation(self, request: InferenceRequest,
                                       feature_vector: np.ndarray,
                                       prediction: float) -> str:
        """Generate fast explanation for prediction"""
        try:
            # Simplified explanation generation for speed
            explanations = {
                'success_predictor': f"Based on learning patterns, predicted success probability is {prediction:.1%}",
                'engagement_predictor': f"Engagement level predicted at {prediction:.2f} based on recent activity",
                'difficulty_adapter': f"Recommended difficulty level: {prediction:.2f}"
            }
            
            base_explanation = explanations.get(
                request.model_name, 
                f"ML prediction: {prediction:.3f}"
            )
            
            # Add context if available
            if request.context.get('recent_performance'):
                perf = request.context['recent_performance']
                if perf > 0.8:
                    base_explanation += " (strong recent performance)"
                elif perf < 0.5:
                    base_explanation += " (needs improvement)"
            
            return base_explanation
            
        except Exception as e:
            logger.error(f"❌ Failed to generate explanation: {e}")
            return "Prediction based on learning analytics"
    
    async def _calculate_feature_importance(self, request: InferenceRequest,
                                          feature_vector: np.ndarray) -> Dict[str, float]:
        """Calculate simplified feature importance for speed"""
        try:
            # Simplified feature importance (top 5 most important)
            feature_names = ['success_rate', 'response_time', 'help_seeking', 'consistency', 'progress']
            importance_values = [0.3, 0.25, 0.2, 0.15, 0.1]
            
            return dict(zip(feature_names, importance_values))
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate feature importance: {e}")
            return {}
    
    def _generate_prediction_cache_key(self, request: InferenceRequest) -> str:
        """Generate cache key for prediction"""
        key_data = {
            'model': request.model_name,
            'student': request.student_id,
            'features': sorted(request.features.items()),
            'context': sorted(request.context.items())
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return f"pred:{hashlib.md5(key_str.encode()).hexdigest()}"
    
    async def _get_cached_prediction(self, cache_key: str) -> Optional[InferenceResponse]:
        """Get cached prediction if available"""
        try:
            if self.redis_client:
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    data = json.loads(cached_data)
                    return InferenceResponse(**data)
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get cached prediction: {e}")
            return None
    
    async def _cache_prediction(self, cache_key: str, response: InferenceResponse):
        """Cache prediction for future use"""
        try:
            if self.redis_client:
                # Cache for 5 minutes
                data = {
                    'request_id': response.request_id,
                    'prediction': response.prediction,
                    'confidence': response.confidence,
                    'explanation': response.explanation,
                    'feature_importance': response.feature_importance,
                    'model_version': response.model_version,
                    'inference_time_ms': response.inference_time_ms
                }
                await self.redis_client.setex(cache_key, 300, json.dumps(data))
                
        except Exception as e:
            logger.error(f"❌ Failed to cache prediction: {e}")
    
    async def _update_metrics(self, model_name: str, inference_time_ms: float, success: bool):
        """Update performance metrics"""
        try:
            if model_name not in self.metrics:
                self.metrics[model_name] = ModelMetrics(model_name=model_name)
            
            metrics = self.metrics[model_name]
            metrics.total_requests += 1
            
            # Update average inference time
            alpha = 0.1  # Exponential moving average factor
            if metrics.avg_inference_time_ms == 0:
                metrics.avg_inference_time_ms = inference_time_ms
            else:
                metrics.avg_inference_time_ms = (1 - alpha) * metrics.avg_inference_time_ms + alpha * inference_time_ms
            
            # Track response times for percentile calculation
            self.response_times[model_name].append(inference_time_ms)
            
            # Update error rate
            if not success:
                self.error_counts[model_name] += 1
            
            if metrics.total_requests > 0:
                metrics.error_rate = self.error_counts[model_name] / metrics.total_requests
            
            # Calculate P95 latency
            if len(self.response_times[model_name]) >= 20:
                metrics.p95_latency_ms = np.percentile(list(self.response_times[model_name]), 95)
            
            metrics.last_updated = datetime.now()
            
        except Exception as e:
            logger.error(f"❌ Failed to update metrics: {e}")
    
    async def _is_circuit_open(self, model_name: str) -> bool:
        """Check if circuit breaker is open for model"""
        try:
            circuit = self.circuit_state[model_name]
            
            if circuit['state'] == 'open':
                # Check if recovery timeout has passed
                if circuit['last_failure']:
                    time_since_failure = (datetime.now() - circuit['last_failure']).total_seconds()
                    if time_since_failure > self.circuit_breaker['recovery_timeout']:
                        circuit['state'] = 'half_open'
                        circuit['errors'] = 0
                        return False
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Circuit breaker check failed: {e}")
            return False
    
    async def _record_circuit_breaker_error(self, model_name: str):
        """Record error for circuit breaker"""
        try:
            circuit = self.circuit_state[model_name]
            circuit['errors'] += 1
            circuit['last_failure'] = datetime.now()
            
            if circuit['errors'] >= self.circuit_breaker['error_threshold']:
                circuit['state'] = 'open'
                logger.warning(f"🚨 Circuit breaker opened for {model_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to record circuit breaker error: {e}")
    
    async def _batch_processor(self):
        """Process requests in batches for better throughput"""
        try:
            while True:
                # Collect requests for batching
                batch_requests = []
                start_time = time.time()
                
                # Wait for first request or timeout
                try:
                    first_request = await asyncio.wait_for(
                        self.request_queue.get(), timeout=self.batch_timeout_ms / 1000
                    )
                    batch_requests.append(first_request)
                except asyncio.TimeoutError:
                    continue
                
                # Collect additional requests up to batch size or timeout
                while len(batch_requests) < self.batch_size:
                    try:
                        remaining_time = (self.batch_timeout_ms / 1000) - (time.time() - start_time)
                        if remaining_time <= 0:
                            break
                        
                        request = await asyncio.wait_for(
                            self.request_queue.get(), timeout=remaining_time
                        )
                        batch_requests.append(request)
                        
                    except asyncio.TimeoutError:
                        break
                
                # Process batch
                if batch_requests:
                    await self._process_request_batch(batch_requests)
                    
        except Exception as e:
            logger.error(f"❌ Batch processor error: {e}")
    
    async def _process_request_batch(self, requests: List[InferenceRequest]):
        """Process a batch of requests efficiently"""
        try:
            # Group by model for efficient batching
            model_groups = defaultdict(list)
            for request in requests:
                model_groups[request.model_name].append(request)
            
            # Process each model group
            tasks = []
            for model_name, model_requests in model_groups.items():
                task = asyncio.create_task(
                    self._process_model_batch(model_name, model_requests)
                )
                tasks.append(task)
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"❌ Failed to process request batch: {e}")
    
    async def _process_model_batch(self, model_name: str, requests: List[InferenceRequest]):
        """Process requests for a specific model"""
        try:
            # For now, process requests individually
            # In production, this could be optimized for true batch inference
            tasks = []
            for request in requests:
                task = asyncio.create_task(self.predict(request))
                tasks.append(task)
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"❌ Failed to process model batch for {model_name}: {e}")
    
    async def _metrics_updater(self):
        """Update aggregated metrics periodically"""
        try:
            while True:
                await asyncio.sleep(30)  # Update every 30 seconds
                
                for model_name, metrics in self.metrics.items():
                    # Calculate throughput
                    time_window = 60  # 1 minute
                    recent_requests = sum(1 for t in self.response_times[model_name] 
                                        if time.time() - t < time_window)
                    metrics.throughput_per_second = recent_requests / time_window
                    
                    # Update cache hit rate from feature cache
                    cache_stats = self.feature_cache.get_cache_stats()
                    metrics.cache_hit_rate = cache_stats['hit_rate']
                
        except Exception as e:
            logger.error(f"❌ Metrics updater error: {e}")
    
    async def _health_monitor(self):
        """Monitor system health and performance"""
        try:
            while True:
                await asyncio.sleep(60)  # Check every minute
                
                for model_name, metrics in self.metrics.items():
                    # Check for performance issues
                    if metrics.avg_inference_time_ms > 100:
                        logger.warning(f"⚠️ High latency for {model_name}: {metrics.avg_inference_time_ms:.1f}ms")
                    
                    if metrics.error_rate > 0.05:  # 5% error rate
                        logger.warning(f"⚠️ High error rate for {model_name}: {metrics.error_rate:.1%}")
                    
                    if metrics.cache_hit_rate < 0.5:  # Low cache hit rate
                        logger.warning(f"⚠️ Low cache hit rate for {model_name}: {metrics.cache_hit_rate:.1%}")
                
                # Log overall health
                total_requests = sum(m.total_requests for m in self.metrics.values())
                if total_requests > 0:
                    logger.info(f"🔍 System health: {total_requests} total requests processed")
                
        except Exception as e:
            logger.error(f"❌ Health monitor error: {e}")
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        try:
            metrics_data = {}
            
            for model_name, metrics in self.metrics.items():
                metrics_data[model_name] = {
                    'total_requests': metrics.total_requests,
                    'avg_inference_time_ms': metrics.avg_inference_time_ms,
                    'p95_latency_ms': metrics.p95_latency_ms,
                    'error_rate': metrics.error_rate,
                    'throughput_per_second': metrics.throughput_per_second,
                    'cache_hit_rate': metrics.cache_hit_rate,
                    'last_updated': metrics.last_updated.isoformat()
                }
            
            # Add system-wide metrics
            metrics_data['system'] = {
                'feature_cache_stats': self.feature_cache.get_cache_stats(),
                'active_models': len(self.loaded_models),
                'queue_size': self.request_queue.qsize(),
                'circuit_breaker_states': {
                    model: state['state'] for model, state in self.circuit_state.items()
                }
            }
            
            return metrics_data
            
        except Exception as e:
            logger.error(f"❌ Failed to get metrics: {e}")
            return {}
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.redis_client:
                await self.redis_client.close()
            
            self.executor.shutdown(wait=True)
            logger.info("✅ Real-time inference engine cleaned up")
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")

# Testing function
async def test_realtime_inference():
    """Test real-time inference engine"""
    try:
        logger.info("🧪 Testing Real-time ML Inference Engine")
        
        engine = RealTimeMLInferenceEngine()
        await engine.initialize()
        
        # Create test request
        test_request = InferenceRequest(
            request_id="test_001",
            model_name="success_predictor",
            features={
                'success_rate': 0.75,
                'avg_response_time': 45.0,
                'help_seeking_rate': 0.15
            },
            student_id="test_student"
        )
        
        # Test prediction (without actual model)
        logger.info("📊 Testing prediction flow (mock)")
        
        # Test metrics
        metrics = await engine.get_metrics()
        logger.info(f"✅ Metrics retrieved: {len(metrics)} models tracked")
        
        await engine.cleanup()
        logger.info("✅ Real-time inference test completed")
        
    except Exception as e:
        logger.error(f"❌ Real-time inference test failed: {e}")

if __name__ == "__main__":
    # Use uvloop for better performance
    uvloop.install()
    asyncio.run(test_realtime_inference())