#!/usr/bin/env python3
"""
Real-Time Prediction Pipeline for Physics Assistant Phase 6.3
Implements streaming analytics, real-time inference, and live prediction updates
with sub-100ms response times and continuous model monitoring.
"""

import asyncio
import json
import logging
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from enum import Enum
import uuid
import redis
import aioredis
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import websockets
import websockets.server
from websockets.exceptions import ConnectionClosed
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
import asyncio
import msgpack
import hashlib

# Import prediction engine
from .predictive_analytics import PredictiveAnalyticsEngine, PredictionResult, EarlyWarningAlert

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionType(Enum):
    SUCCESS_PROBABILITY = "success_probability"
    ENGAGEMENT_LEVEL = "engagement_level"
    TIME_TO_MASTERY = "time_to_mastery"
    LEARNING_OUTCOME = "learning_outcome"
    RISK_ASSESSMENT = "risk_assessment"

class StreamingStatus(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"

@dataclass
class StreamingPredictionRequest:
    """Real-time prediction request"""
    request_id: str
    student_id: str
    prediction_types: List[PredictionType]
    context: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1=highest, 5=lowest
    callback_url: Optional[str] = None
    websocket_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class StreamingPredictionResponse:
    """Real-time prediction response"""
    request_id: str
    student_id: str
    predictions: Dict[str, PredictionResult]
    processing_time_ms: float
    model_versions: Dict[str, str]
    confidence_scores: Dict[str, float]
    alerts: List[EarlyWarningAlert] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class StudentPredictionStream:
    """Live prediction stream for a student"""
    student_id: str
    active_predictions: Dict[PredictionType, PredictionResult]
    prediction_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    last_update: datetime = field(default_factory=datetime.now)
    update_frequency_seconds: int = 30
    websocket_connections: set = field(default_factory=set)
    alert_subscriptions: set = field(default_factory=set)

@dataclass
class PredictionStreamMetrics:
    """Metrics for prediction streaming"""
    total_requests: int = 0
    successful_predictions: int = 0
    failed_predictions: int = 0
    avg_latency_ms: float = 0.0
    predictions_per_second: float = 0.0
    active_streams: int = 0
    websocket_connections: int = 0
    cache_hit_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

class RealtimePredictionCache:
    """High-performance caching for predictions"""
    
    def __init__(self, redis_client: aioredis.Redis, default_ttl: int = 60):
        self.redis = redis_client
        self.default_ttl = default_ttl
        self.local_cache = {}
        self.cache_hits = Counter('prediction_cache_hits_total', 'Prediction cache hits')
        self.cache_misses = Counter('prediction_cache_misses_total', 'Prediction cache misses')
    
    async def get_prediction(self, student_id: str, prediction_type: PredictionType) -> Optional[PredictionResult]:
        """Get cached prediction"""
        try:
            cache_key = f"prediction:{student_id}:{prediction_type.value}"
            
            # Check local cache first
            if cache_key in self.local_cache:
                cached_data, expiry = self.local_cache[cache_key]
                if datetime.now() < expiry:
                    self.cache_hits.inc()
                    return cached_data
                else:
                    del self.local_cache[cache_key]
            
            # Check Redis cache
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                prediction_data = msgpack.unpackb(cached_data, raw=False)
                prediction = PredictionResult(**prediction_data)
                
                # Store in local cache
                self.local_cache[cache_key] = (prediction, datetime.now() + timedelta(seconds=30))
                self.cache_hits.inc()
                return prediction
            
            self.cache_misses.inc()
            return None
            
        except Exception as e:
            logger.error(f"❌ Cache get error: {e}")
            return None
    
    async def set_prediction(self, student_id: str, prediction_type: PredictionType, 
                           prediction: PredictionResult, ttl: Optional[int] = None) -> bool:
        """Cache prediction result"""
        try:
            cache_key = f"prediction:{student_id}:{prediction_type.value}"
            ttl = ttl or self.default_ttl
            
            # Convert to dict for serialization
            prediction_dict = {
                'student_id': prediction.student_id,
                'prediction_type': prediction.prediction_type,
                'predicted_value': prediction.predicted_value,
                'confidence_score': prediction.confidence_score,
                'confidence_interval': prediction.confidence_interval,
                'contributing_factors': prediction.contributing_factors,
                'risk_level': prediction.risk_level,
                'recommendations': prediction.recommendations,
                'model_version': prediction.model_version,
                'prediction_date': prediction.prediction_date.isoformat()
            }
            
            # Store in Redis
            packed_data = msgpack.packb(prediction_dict, use_bin_type=True)
            await self.redis.setex(cache_key, ttl, packed_data)
            
            # Store in local cache
            self.local_cache[cache_key] = (prediction, datetime.now() + timedelta(seconds=30))
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Cache set error: {e}")
            return False

class WebSocketManager:
    """WebSocket connection manager for real-time updates"""
    
    def __init__(self):
        self.connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.student_subscriptions: Dict[str, set] = defaultdict(set)
        self.connection_metrics = Gauge('websocket_connections_active', 'Active WebSocket connections')
    
    async def register_connection(self, websocket: websockets.WebSocketServerProtocol, 
                                connection_id: str, student_id: str):
        """Register new WebSocket connection"""
        try:
            self.connections[connection_id] = websocket
            self.student_subscriptions[student_id].add(connection_id)
            self.connection_metrics.set(len(self.connections))
            
            logger.info(f"🔌 Registered WebSocket connection {connection_id} for student {student_id}")
            
            # Send connection confirmation
            await self.send_to_connection(connection_id, {
                'type': 'connection_confirmed',
                'connection_id': connection_id,
                'student_id': student_id,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Failed to register WebSocket connection: {e}")
    
    async def unregister_connection(self, connection_id: str):
        """Unregister WebSocket connection"""
        try:
            if connection_id in self.connections:
                del self.connections[connection_id]
            
            # Remove from student subscriptions
            for student_id, connections in self.student_subscriptions.items():
                connections.discard(connection_id)
            
            self.connection_metrics.set(len(self.connections))
            logger.info(f"🔌 Unregistered WebSocket connection {connection_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to unregister WebSocket connection: {e}")
    
    async def send_to_connection(self, connection_id: str, message: Dict[str, Any]) -> bool:
        """Send message to specific connection"""
        try:
            if connection_id in self.connections:
                websocket = self.connections[connection_id]
                await websocket.send(json.dumps(message))
                return True
            return False
            
        except ConnectionClosed:
            await self.unregister_connection(connection_id)
            return False
        except Exception as e:
            logger.error(f"❌ Failed to send WebSocket message: {e}")
            return False
    
    async def broadcast_to_student(self, student_id: str, message: Dict[str, Any]) -> int:
        """Broadcast message to all connections for a student"""
        try:
            connections = list(self.student_subscriptions[student_id])
            sent_count = 0
            
            for connection_id in connections:
                if await self.send_to_connection(connection_id, message):
                    sent_count += 1
            
            return sent_count
            
        except Exception as e:
            logger.error(f"❌ Failed to broadcast to student {student_id}: {e}")
            return 0
    
    async def broadcast_alert(self, alert: EarlyWarningAlert) -> int:
        """Broadcast alert to relevant connections"""
        try:
            alert_message = {
                'type': 'early_warning_alert',
                'alert': {
                    'student_id': alert.student_id,
                    'alert_type': alert.alert_type,
                    'severity': alert.severity,
                    'predicted_outcome': alert.predicted_outcome,
                    'confidence': alert.confidence,
                    'triggered_by': alert.triggered_by,
                    'recommended_actions': alert.recommended_actions,
                    'alert_date': alert.alert_date.isoformat()
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return await self.broadcast_to_student(alert.student_id, alert_message)
            
        except Exception as e:
            logger.error(f"❌ Failed to broadcast alert: {e}")
            return 0

class RealtimePredictionPipeline:
    """Real-time prediction pipeline with streaming analytics"""
    
    def __init__(self, prediction_engine: PredictiveAnalyticsEngine, 
                 redis_url: str = "redis://localhost:6379", 
                 websocket_port: int = 8765):
        self.prediction_engine = prediction_engine
        self.redis_url = redis_url
        self.websocket_port = websocket_port
        
        # Components
        self.redis_client = None
        self.prediction_cache = None
        self.websocket_manager = WebSocketManager()
        
        # Streaming configuration
        self.student_streams: Dict[str, StudentPredictionStream] = {}
        self.prediction_queues: Dict[PredictionType, asyncio.Queue] = {}
        self.processing_active = False
        
        # Performance metrics
        self.metrics = PredictionStreamMetrics()
        self.latency_histogram = Histogram(
            'prediction_latency_seconds',
            'Prediction processing latency',
            ['prediction_type'],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
        )
        self.prediction_counter = Counter(
            'predictions_total',
            'Total predictions made',
            ['prediction_type', 'status']
        )
        
        # Thread pool for CPU-intensive operations
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        # WebSocket server
        self.websocket_server = None
    
    async def initialize(self):
        """Initialize the real-time prediction pipeline"""
        try:
            logger.info("🚀 Initializing Real-Time Prediction Pipeline")
            
            # Initialize Redis connection
            self.redis_client = await aioredis.from_url(self.redis_url, decode_responses=False)
            
            # Initialize prediction cache
            self.prediction_cache = RealtimePredictionCache(self.redis_client)
            
            # Initialize prediction queues
            for prediction_type in PredictionType:
                self.prediction_queues[prediction_type] = asyncio.Queue(maxsize=1000)
            
            # Start processing workers
            self.processing_active = True
            await self._start_processing_workers()
            
            # Start WebSocket server
            await self._start_websocket_server()
            
            # Start monitoring tasks
            await self._start_monitoring_tasks()
            
            logger.info("✅ Real-Time Prediction Pipeline initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Real-Time Prediction Pipeline: {e}")
            return False
    
    async def predict_streaming(self, request: StreamingPredictionRequest) -> StreamingPredictionResponse:
        """Make real-time streaming predictions"""
        start_time = time.time()
        
        try:
            predictions = {}
            model_versions = {}
            confidence_scores = {}
            alerts = []
            
            # Process each prediction type
            for prediction_type in request.prediction_types:
                try:
                    # Check cache first
                    cached_prediction = await self.prediction_cache.get_prediction(
                        request.student_id, prediction_type
                    )
                    
                    if cached_prediction:
                        predictions[prediction_type.value] = cached_prediction
                        model_versions[prediction_type.value] = cached_prediction.model_version
                        confidence_scores[prediction_type.value] = cached_prediction.confidence_score
                    else:
                        # Make new prediction
                        prediction = await self._make_prediction(request.student_id, prediction_type)
                        
                        if prediction:
                            predictions[prediction_type.value] = prediction
                            model_versions[prediction_type.value] = prediction.model_version
                            confidence_scores[prediction_type.value] = prediction.confidence_score
                            
                            # Cache the prediction
                            await self.prediction_cache.set_prediction(
                                request.student_id, prediction_type, prediction
                            )
                            
                            # Check for alerts
                            if prediction.risk_level in ['high', 'critical']:
                                alert = EarlyWarningAlert(
                                    student_id=request.student_id,
                                    alert_type=f"{prediction_type.value}_risk",
                                    severity=prediction.risk_level,
                                    predicted_outcome=f"{prediction.prediction_type}: {prediction.predicted_value:.2%}",
                                    confidence=prediction.confidence_score,
                                    triggered_by=[f"low_{prediction_type.value}"],
                                    recommended_actions=prediction.recommendations
                                )
                                alerts.append(alert)
                    
                    # Update metrics
                    self.prediction_counter.labels(
                        prediction_type=prediction_type.value,
                        status='success'
                    ).inc()
                    
                except Exception as e:
                    logger.error(f"❌ Failed to make {prediction_type.value} prediction: {e}")
                    self.prediction_counter.labels(
                        prediction_type=prediction_type.value,
                        status='error'
                    ).inc()
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Record latency
            for prediction_type in request.prediction_types:
                self.latency_histogram.labels(
                    prediction_type=prediction_type.value
                ).observe(processing_time_ms / 1000.0)
            
            response = StreamingPredictionResponse(
                request_id=request.request_id,
                student_id=request.student_id,
                predictions=predictions,
                processing_time_ms=processing_time_ms,
                model_versions=model_versions,
                confidence_scores=confidence_scores,
                alerts=alerts
            )
            
            # Update student stream
            await self._update_student_stream(request.student_id, predictions, alerts)
            
            # Send WebSocket updates
            if request.websocket_id:
                await self._send_websocket_update(request.websocket_id, response)
            
            # Broadcast alerts
            for alert in alerts:
                await self.websocket_manager.broadcast_alert(alert)
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Streaming prediction error: {e}")
            processing_time_ms = (time.time() - start_time) * 1000
            
            return StreamingPredictionResponse(
                request_id=request.request_id,
                student_id=request.student_id,
                predictions={},
                processing_time_ms=processing_time_ms,
                model_versions={},
                confidence_scores={}
            )
    
    async def _make_prediction(self, student_id: str, prediction_type: PredictionType) -> Optional[PredictionResult]:
        """Make individual prediction based on type"""
        try:
            if prediction_type == PredictionType.SUCCESS_PROBABILITY:
                return await self.prediction_engine.predict_student_success(student_id)
            elif prediction_type == PredictionType.ENGAGEMENT_LEVEL:
                return await self.prediction_engine.predict_student_engagement(student_id)
            elif prediction_type == PredictionType.TIME_TO_MASTERY:
                return await self._predict_time_to_mastery(student_id)
            elif prediction_type == PredictionType.LEARNING_OUTCOME:
                return await self._predict_learning_outcome(student_id)
            elif prediction_type == PredictionType.RISK_ASSESSMENT:
                return await self._predict_overall_risk(student_id)
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to make {prediction_type.value} prediction: {e}")
            return None
    
    async def _predict_time_to_mastery(self, student_id: str) -> PredictionResult:
        """Predict time to mastery for current concepts"""
        try:
            # Get student features
            features = await self.prediction_engine.extract_student_features(student_id)
            
            if not features:
                raise ValueError(f"No features available for student {student_id}")
            
            # Calculate learning velocity and current mastery
            learning_velocity = features.get('learning_velocity', 0.1)
            current_success_rate = features.get('overall_success_rate', 0.5)
            concept_coverage = features.get('concept_coverage', 1)
            
            # Estimate time to mastery (80% success rate) based on current trajectory
            mastery_threshold = 0.8
            if current_success_rate >= mastery_threshold:
                estimated_days = 0  # Already mastered
            elif learning_velocity > 0:
                improvement_needed = mastery_threshold - current_success_rate
                estimated_days = improvement_needed / learning_velocity
            else:
                estimated_days = 30 * concept_coverage  # Fallback estimate
            
            # Apply concept difficulty multiplier
            difficulty_multiplier = min(2.0, concept_coverage * 0.5)
            estimated_days *= difficulty_multiplier
            
            # Cap at reasonable bounds
            estimated_days = max(1, min(90, estimated_days))
            
            confidence = min(0.9, 0.5 + abs(learning_velocity) * 2)
            
            return PredictionResult(
                student_id=student_id,
                prediction_type='time_to_mastery',
                predicted_value=estimated_days,
                confidence_score=confidence,
                confidence_interval=(estimated_days * 0.8, estimated_days * 1.2),
                contributing_factors={
                    'learning_velocity': learning_velocity,
                    'current_success_rate': current_success_rate,
                    'concept_coverage': concept_coverage
                },
                risk_level='low' if estimated_days <= 14 else 'medium' if estimated_days <= 30 else 'high',
                recommendations=[
                    f"Estimated {estimated_days:.0f} days to mastery",
                    "Continue consistent practice" if estimated_days <= 14 else "Increase practice frequency",
                    "Focus on weak areas" if current_success_rate < 0.6 else "Maintain current pace"
                ],
                model_version='1.0'
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to predict time to mastery: {e}")
            raise
    
    async def _predict_learning_outcome(self, student_id: str) -> PredictionResult:
        """Predict learning outcome for current course"""
        try:
            # Get multiple prediction types for comprehensive assessment
            success_prediction = await self.prediction_engine.predict_student_success(student_id)
            engagement_prediction = await self.prediction_engine.predict_student_engagement(student_id)
            
            # Combine predictions for overall outcome
            success_weight = 0.6
            engagement_weight = 0.4
            
            combined_score = (
                success_prediction.predicted_value * success_weight +
                engagement_prediction.predicted_value * engagement_weight
            )
            
            # Determine outcome category
            if combined_score >= 0.85:
                outcome = "excellent"
            elif combined_score >= 0.75:
                outcome = "good"
            elif combined_score >= 0.65:
                outcome = "satisfactory"
            elif combined_score >= 0.5:
                outcome = "needs_improvement"
            else:
                outcome = "at_risk"
            
            confidence = (success_prediction.confidence_score + engagement_prediction.confidence_score) / 2
            
            return PredictionResult(
                student_id=student_id,
                prediction_type='learning_outcome',
                predicted_value=combined_score,
                confidence_score=confidence,
                confidence_interval=(combined_score * 0.9, min(1.0, combined_score * 1.1)),
                contributing_factors={
                    'success_probability': success_prediction.predicted_value,
                    'engagement_level': engagement_prediction.predicted_value,
                    'combined_assessment': combined_score
                },
                risk_level='low' if outcome in ['excellent', 'good'] else 'medium' if outcome == 'satisfactory' else 'high',
                recommendations=[
                    f"Predicted outcome: {outcome}",
                    *success_prediction.recommendations[:2],
                    *engagement_prediction.recommendations[:2]
                ],
                model_version='1.0'
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to predict learning outcome: {e}")
            raise
    
    async def _predict_overall_risk(self, student_id: str) -> PredictionResult:
        """Predict overall risk assessment"""
        try:
            # Get comprehensive risk indicators
            features = await self.prediction_engine.extract_student_features(student_id)
            
            if not features:
                raise ValueError(f"No features available for student {student_id}")
            
            # Calculate risk factors
            risk_factors = {}
            
            # Performance risk
            success_rate = features.get('overall_success_rate', 0.5)
            risk_factors['performance_risk'] = 1.0 - success_rate
            
            # Engagement risk
            interaction_frequency = features.get('interactions_per_day', 5)
            risk_factors['engagement_risk'] = max(0, 1.0 - interaction_frequency / 10.0)
            
            # Consistency risk
            consistency = features.get('study_consistency', 0.5)
            risk_factors['consistency_risk'] = 1.0 - consistency
            
            # Help-seeking risk (too much or too little)
            help_seeking_rate = features.get('help_seeking_rate', 0.2)
            optimal_help_rate = 0.2
            risk_factors['help_seeking_risk'] = abs(help_seeking_rate - optimal_help_rate) / optimal_help_rate
            
            # Response time risk
            avg_response_time = features.get('avg_response_time', 5000)  # milliseconds
            risk_factors['response_time_risk'] = min(1.0, max(0, (avg_response_time - 3000) / 10000))
            
            # Calculate overall risk score
            weights = {
                'performance_risk': 0.4,
                'engagement_risk': 0.25,
                'consistency_risk': 0.15,
                'help_seeking_risk': 0.1,
                'response_time_risk': 0.1
            }
            
            overall_risk = sum(risk_factors[factor] * weights[factor] for factor in risk_factors)
            overall_risk = min(1.0, max(0.0, overall_risk))
            
            # Determine risk level
            if overall_risk >= 0.7:
                risk_level = 'critical'
            elif overall_risk >= 0.5:
                risk_level = 'high'
            elif overall_risk >= 0.3:
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            # Generate recommendations based on highest risk factors
            recommendations = []
            sorted_risks = sorted(risk_factors.items(), key=lambda x: x[1], reverse=True)
            
            for risk_factor, risk_value in sorted_risks[:3]:
                if risk_value > 0.3:
                    if risk_factor == 'performance_risk':
                        recommendations.append("Focus on improving problem-solving accuracy")
                    elif risk_factor == 'engagement_risk':
                        recommendations.append("Increase study frequency and interaction")
                    elif risk_factor == 'consistency_risk':
                        recommendations.append("Establish more consistent study schedule")
                    elif risk_factor == 'help_seeking_risk':
                        recommendations.append("Balance independent work with seeking help")
                    elif risk_factor == 'response_time_risk':
                        recommendations.append("Work on problem-solving efficiency")
            
            if not recommendations:
                recommendations.append("Continue current learning approach")
            
            confidence = 0.8  # High confidence in risk assessment
            
            return PredictionResult(
                student_id=student_id,
                prediction_type='risk_assessment',
                predicted_value=overall_risk,
                confidence_score=confidence,
                confidence_interval=(overall_risk * 0.9, min(1.0, overall_risk * 1.1)),
                contributing_factors=risk_factors,
                risk_level=risk_level,
                recommendations=recommendations,
                model_version='1.0'
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to predict overall risk: {e}")
            raise
    
    async def _update_student_stream(self, student_id: str, predictions: Dict[str, PredictionResult], 
                                   alerts: List[EarlyWarningAlert]):
        """Update student prediction stream"""
        try:
            if student_id not in self.student_streams:
                self.student_streams[student_id] = StudentPredictionStream(student_id=student_id)
            
            stream = self.student_streams[student_id]
            
            # Update active predictions
            for prediction_type_str, prediction in predictions.items():
                prediction_type = PredictionType(prediction_type_str)
                stream.active_predictions[prediction_type] = prediction
            
            # Add to prediction history
            stream.prediction_history.append({
                'timestamp': datetime.now(),
                'predictions': predictions,
                'alerts': alerts
            })
            
            stream.last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"❌ Failed to update student stream: {e}")
    
    async def _send_websocket_update(self, websocket_id: str, response: StreamingPredictionResponse):
        """Send prediction update via WebSocket"""
        try:
            message = {
                'type': 'prediction_update',
                'data': {
                    'request_id': response.request_id,
                    'student_id': response.student_id,
                    'predictions': {
                        pred_type: {
                            'predicted_value': pred.predicted_value,
                            'confidence_score': pred.confidence_score,
                            'risk_level': pred.risk_level,
                            'recommendations': pred.recommendations
                        } for pred_type, pred in response.predictions.items()
                    },
                    'processing_time_ms': response.processing_time_ms,
                    'timestamp': response.timestamp.isoformat()
                }
            }
            
            await self.websocket_manager.send_to_connection(websocket_id, message)
            
        except Exception as e:
            logger.error(f"❌ Failed to send WebSocket update: {e}")
    
    async def _start_processing_workers(self):
        """Start background processing workers"""
        try:
            # Start workers for each prediction type
            for prediction_type in PredictionType:
                asyncio.create_task(self._prediction_worker(prediction_type))
            
            # Start stream update worker
            asyncio.create_task(self._stream_update_worker())
            
        except Exception as e:
            logger.error(f"❌ Failed to start processing workers: {e}")
    
    async def _prediction_worker(self, prediction_type: PredictionType):
        """Process prediction requests for specific type"""
        try:
            queue = self.prediction_queues[prediction_type]
            
            while self.processing_active:
                try:
                    # Get next request with timeout
                    request = await asyncio.wait_for(queue.get(), timeout=1.0)
                    
                    # Process prediction
                    await self.predict_streaming(request)
                    
                    # Mark task as done
                    queue.task_done()
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"❌ Prediction worker {prediction_type.value} error: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Prediction worker {prediction_type.value} fatal error: {e}")
    
    async def _stream_update_worker(self):
        """Continuously update active prediction streams"""
        try:
            while self.processing_active:
                try:
                    current_time = datetime.now()
                    
                    for student_id, stream in list(self.student_streams.items()):
                        # Check if stream needs update
                        time_since_update = (current_time - stream.last_update).total_seconds()
                        
                        if time_since_update >= stream.update_frequency_seconds:
                            # Update predictions for this student
                            request = StreamingPredictionRequest(
                                request_id=str(uuid.uuid4()),
                                student_id=student_id,
                                prediction_types=list(PredictionType),
                                priority=3  # Lower priority for automatic updates
                            )
                            
                            await self.predict_streaming(request)
                    
                    await asyncio.sleep(10)  # Check every 10 seconds
                    
                except Exception as e:
                    logger.error(f"❌ Stream update worker error: {e}")
                    await asyncio.sleep(10)
                    
        except Exception as e:
            logger.error(f"❌ Stream update worker fatal error: {e}")
    
    async def _start_websocket_server(self):
        """Start WebSocket server for real-time updates"""
        try:
            async def handle_websocket(websocket, path):
                connection_id = str(uuid.uuid4())
                student_id = None
                
                try:
                    # Wait for client identification
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    if data.get('type') == 'identify':
                        student_id = data.get('student_id')
                        if student_id:
                            await self.websocket_manager.register_connection(
                                websocket, connection_id, student_id
                            )
                            
                            # Send current predictions if available
                            if student_id in self.student_streams:
                                stream = self.student_streams[student_id]
                                current_predictions = {
                                    pred_type.value: pred for pred_type, pred in stream.active_predictions.items()
                                }
                                
                                if current_predictions:
                                    await self.websocket_manager.send_to_connection(connection_id, {
                                        'type': 'initial_predictions',
                                        'predictions': current_predictions,
                                        'timestamp': datetime.now().isoformat()
                                    })
                    
                    # Keep connection alive
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            
                            if data.get('type') == 'prediction_request':
                                # Handle real-time prediction request
                                request = StreamingPredictionRequest(
                                    request_id=str(uuid.uuid4()),
                                    student_id=student_id,
                                    prediction_types=[PredictionType(pt) for pt in data.get('prediction_types', [])],
                                    websocket_id=connection_id
                                )
                                
                                await self.predict_streaming(request)
                                
                        except json.JSONDecodeError:
                            await websocket.send(json.dumps({
                                'type': 'error',
                                'message': 'Invalid JSON format'
                            }))
                            
                except ConnectionClosed:
                    pass
                except Exception as e:
                    logger.error(f"❌ WebSocket handler error: {e}")
                finally:
                    await self.websocket_manager.unregister_connection(connection_id)
            
            # Start WebSocket server
            self.websocket_server = await websockets.serve(
                handle_websocket, "localhost", self.websocket_port
            )
            
            logger.info(f"🔌 WebSocket server started on port {self.websocket_port}")
            
        except Exception as e:
            logger.error(f"❌ Failed to start WebSocket server: {e}")
    
    async def _start_monitoring_tasks(self):
        """Start monitoring and metrics collection"""
        try:
            asyncio.create_task(self._metrics_collector())
            asyncio.create_task(self._performance_monitor())
            
        except Exception as e:
            logger.error(f"❌ Failed to start monitoring tasks: {e}")
    
    async def _metrics_collector(self):
        """Collect and update performance metrics"""
        try:
            while self.processing_active:
                try:
                    # Update metrics
                    self.metrics.active_streams = len(self.student_streams)
                    self.metrics.websocket_connections = len(self.websocket_manager.connections)
                    self.metrics.last_updated = datetime.now()
                    
                    await asyncio.sleep(30)  # Update every 30 seconds
                    
                except Exception as e:
                    logger.error(f"❌ Metrics collector error: {e}")
                    await asyncio.sleep(30)
                    
        except Exception as e:
            logger.error(f"❌ Metrics collector fatal error: {e}")
    
    async def _performance_monitor(self):
        """Monitor system performance and auto-scale"""
        try:
            while self.processing_active:
                try:
                    # Check queue depths
                    total_queue_depth = sum(queue.qsize() for queue in self.prediction_queues.values())
                    
                    if total_queue_depth > 500:  # High load
                        logger.warning(f"⚠️ High prediction queue depth: {total_queue_depth}")
                    
                    # Check processing times
                    avg_latency = self.metrics.avg_latency_ms
                    if avg_latency > 100:  # Target <100ms
                        logger.warning(f"⚠️ High prediction latency: {avg_latency:.1f}ms")
                    
                    await asyncio.sleep(60)  # Monitor every minute
                    
                except Exception as e:
                    logger.error(f"❌ Performance monitor error: {e}")
                    await asyncio.sleep(60)
                    
        except Exception as e:
            logger.error(f"❌ Performance monitor fatal error: {e}")
    
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        try:
            return {
                'processing_active': self.processing_active,
                'active_streams': len(self.student_streams),
                'websocket_connections': len(self.websocket_manager.connections),
                'queue_depths': {
                    pred_type.value: queue.qsize() 
                    for pred_type, queue in self.prediction_queues.items()
                },
                'metrics': {
                    'total_requests': self.metrics.total_requests,
                    'successful_predictions': self.metrics.successful_predictions,
                    'avg_latency_ms': self.metrics.avg_latency_ms,
                    'cache_hit_rate': self.metrics.cache_hit_rate
                },
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get pipeline status: {e}")
            return {'error': str(e)}
    
    async def shutdown(self):
        """Gracefully shutdown the pipeline"""
        try:
            logger.info("🛑 Shutting down Real-Time Prediction Pipeline")
            
            self.processing_active = False
            
            # Close WebSocket server
            if self.websocket_server:
                self.websocket_server.close()
                await self.websocket_server.wait_closed()
            
            # Wait for queues to empty
            for queue in self.prediction_queues.values():
                await queue.join()
            
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("✅ Real-Time Prediction Pipeline shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Pipeline shutdown error: {e}")

# Testing function
async def test_realtime_prediction_pipeline():
    """Test the real-time prediction pipeline"""
    try:
        logger.info("🧪 Testing Real-Time Prediction Pipeline")
        
        # Create mock prediction engine
        from .predictive_analytics import PredictiveAnalyticsEngine
        prediction_engine = PredictiveAnalyticsEngine()
        await prediction_engine.initialize()
        
        # Create pipeline
        pipeline = RealtimePredictionPipeline(prediction_engine)
        await pipeline.initialize()
        
        # Test streaming prediction
        request = StreamingPredictionRequest(
            request_id=str(uuid.uuid4()),
            student_id="test_student",
            prediction_types=[PredictionType.SUCCESS_PROBABILITY, PredictionType.ENGAGEMENT_LEVEL],
            context={'test': True}
        )
        
        # Note: This would require database connection for full testing
        # response = await pipeline.predict_streaming(request)
        # logger.info(f"✅ Streaming prediction completed: {len(response.predictions)} predictions")
        
        # Test pipeline status
        status = await pipeline.get_pipeline_status()
        logger.info(f"✅ Pipeline status: {status['processing_active']}")
        
        await pipeline.shutdown()
        logger.info("✅ Real-Time Prediction Pipeline test completed")
        
    except Exception as e:
        logger.error(f"❌ Real-Time Prediction Pipeline test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_realtime_prediction_pipeline())