#!/usr/bin/env python3
"""
Real-time Streaming Analytics Engine - Phase 6
Processes educational interaction streams in real-time for immediate insights,
adaptive learning adjustments, and live dashboard updates.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, AsyncGenerator
from enum import Enum
import uuid
import redis
import aioredis
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import warnings

# Event processing
import asyncio
from asyncio import Queue, Event

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EventType(Enum):
    INTERACTION = "interaction"
    LOGIN = "login"
    LOGOUT = "logout"
    PROBLEM_ATTEMPT = "problem_attempt"
    HELP_REQUEST = "help_request"
    CONCEPT_ACCESS = "concept_access"
    ASSESSMENT_COMPLETION = "assessment_completion"
    RECOMMENDATION_VIEW = "recommendation_view"
    NAVIGATION = "navigation"
    ERROR = "error"

class StreamingMetric(Enum):
    ACTIVE_USERS = "active_users"
    ENGAGEMENT_RATE = "engagement_rate"
    SUCCESS_RATE = "success_rate"
    HELP_REQUEST_RATE = "help_request_rate"
    AVERAGE_SESSION_TIME = "average_session_time"
    CONCEPT_DIFFICULTY = "concept_difficulty"
    REAL_TIME_PERFORMANCE = "real_time_performance"

class WindowType(Enum):
    TUMBLING = "tumbling"     # Non-overlapping fixed windows
    SLIDING = "sliding"       # Overlapping fixed windows
    SESSION = "session"       # Session-based windows

@dataclass
class StreamEvent:
    """Individual stream event"""
    event_id: str
    event_type: EventType
    timestamp: datetime
    user_id: str
    session_id: str
    
    # Event data
    data: Dict[str, Any]
    
    # Context
    user_context: Dict[str, Any] = field(default_factory=dict)
    session_context: Dict[str, Any] = field(default_factory=dict)
    
    # Processing metadata
    processed_at: Optional[datetime] = None
    processing_latency_ms: Optional[float] = None

@dataclass
class StreamWindow:
    """Time window for aggregations"""
    window_id: str
    window_type: WindowType
    start_time: datetime
    end_time: datetime
    duration_seconds: int
    
    # Aggregated data
    event_count: int = 0
    unique_users: set = field(default_factory=set)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Window state
    is_closed: bool = False
    closed_at: Optional[datetime] = None

@dataclass
class RealTimeAlert:
    """Real-time alert triggered by streaming data"""
    alert_id: str
    alert_type: str
    severity: str
    message: str
    triggered_at: datetime
    
    # Context
    affected_users: List[str]
    affected_concepts: List[str]
    metric_values: Dict[str, Any]
    
    # Actions taken
    auto_actions_triggered: List[str] = field(default_factory=list)

class EventProcessor:
    """Process individual events in real-time"""
    
    def __init__(self):
        self.event_handlers = {}
        self.processing_stats = defaultdict(int)
        self.error_rate = deque(maxlen=1000)
    
    def register_handler(self, event_type: EventType, handler: Callable):
        """Register event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    async def process_event(self, event: StreamEvent) -> Dict[str, Any]:
        """Process a single event"""
        start_time = time.time()
        processing_results = {}
        
        try:
            # Mark processing start
            event.processed_at = datetime.now()
            
            # Get handlers for this event type
            handlers = self.event_handlers.get(event.event_type, [])
            
            # Process with each handler
            for handler in handlers:
                try:
                    handler_name = handler.__name__
                    result = await handler(event)
                    processing_results[handler_name] = result
                    
                except Exception as e:
                    logger.error(f"❌ Handler {handler.__name__} failed for event {event.event_id}: {e}")
                    processing_results[handler.__name__] = {'error': str(e)}
            
            # Calculate processing latency
            processing_time = (time.time() - start_time) * 1000
            event.processing_latency_ms = processing_time
            
            # Update stats
            self.processing_stats['events_processed'] += 1
            self.processing_stats['total_processing_time_ms'] += processing_time
            
            # Track success
            self.error_rate.append(0)
            
            return processing_results
            
        except Exception as e:
            # Track error
            self.error_rate.append(1)
            self.processing_stats['processing_errors'] += 1
            
            logger.error(f"❌ Event processing failed for {event.event_id}: {e}")
            return {'error': str(e)}
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get event processing statistics"""
        total_events = self.processing_stats['events_processed']
        total_time = self.processing_stats['total_processing_time_ms']
        
        return {
            'total_events_processed': total_events,
            'average_processing_time_ms': total_time / max(1, total_events),
            'error_rate': np.mean(self.error_rate) if self.error_rate else 0.0,
            'events_per_second': total_events / max(1, total_time / 1000) if total_time > 0 else 0,
            'total_errors': self.processing_stats['processing_errors']
        }

class WindowManager:
    """Manage time windows for aggregations"""
    
    def __init__(self):
        self.active_windows = {}
        self.completed_windows = deque(maxlen=1000)
        self.window_configs = {}
    
    def configure_window(self, metric_name: str, window_type: WindowType,
                        duration_seconds: int, slide_interval_seconds: int = None):
        """Configure a time window"""
        self.window_configs[metric_name] = {
            'window_type': window_type,
            'duration_seconds': duration_seconds,
            'slide_interval_seconds': slide_interval_seconds or duration_seconds
        }
    
    async def process_event_for_windows(self, event: StreamEvent):
        """Process event for all configured windows"""
        try:
            current_time = event.timestamp
            
            for metric_name, config in self.window_configs.items():
                await self._add_event_to_windows(event, metric_name, config, current_time)
            
            # Clean up expired windows
            await self._cleanup_expired_windows(current_time)
            
        except Exception as e:
            logger.error(f"❌ Window processing failed for event {event.event_id}: {e}")
    
    async def _add_event_to_windows(self, event: StreamEvent, metric_name: str,
                                  config: Dict[str, Any], current_time: datetime):
        """Add event to appropriate windows"""
        try:
            window_type = config['window_type']
            duration = config['duration_seconds']
            
            if window_type == WindowType.TUMBLING:
                # Create tumbling windows
                window_start = self._get_tumbling_window_start(current_time, duration)
                window_id = f"{metric_name}_{window_start.timestamp()}"
                
                if window_id not in self.active_windows:
                    self.active_windows[window_id] = StreamWindow(
                        window_id=window_id,
                        window_type=window_type,
                        start_time=window_start,
                        end_time=window_start + timedelta(seconds=duration),
                        duration_seconds=duration
                    )
                
                await self._add_event_to_window(event, self.active_windows[window_id])
                
            elif window_type == WindowType.SLIDING:
                # Create sliding windows
                slide_interval = config['slide_interval_seconds']
                
                # Find all sliding windows this event belongs to
                window_starts = self._get_sliding_window_starts(current_time, duration, slide_interval)
                
                for window_start in window_starts:
                    window_id = f"{metric_name}_{window_start.timestamp()}"
                    
                    if window_id not in self.active_windows:
                        self.active_windows[window_id] = StreamWindow(
                            window_id=window_id,
                            window_type=window_type,
                            start_time=window_start,
                            end_time=window_start + timedelta(seconds=duration),
                            duration_seconds=duration
                        )
                    
                    await self._add_event_to_window(event, self.active_windows[window_id])
            
        except Exception as e:
            logger.error(f"❌ Failed to add event to windows: {e}")
    
    async def _add_event_to_window(self, event: StreamEvent, window: StreamWindow):
        """Add event to a specific window"""
        try:
            # Basic aggregations
            window.event_count += 1
            window.unique_users.add(event.user_id)
            
            # Event-specific aggregations
            if event.event_type == EventType.PROBLEM_ATTEMPT:
                if 'success_attempts' not in window.metrics:
                    window.metrics['success_attempts'] = 0
                    window.metrics['total_attempts'] = 0
                
                window.metrics['total_attempts'] += 1
                if event.data.get('success', False):
                    window.metrics['success_attempts'] += 1
            
            elif event.event_type == EventType.HELP_REQUEST:
                if 'help_requests' not in window.metrics:
                    window.metrics['help_requests'] = 0
                window.metrics['help_requests'] += 1
            
            elif event.event_type == EventType.INTERACTION:
                if 'session_times' not in window.metrics:
                    window.metrics['session_times'] = []
                
                session_time = event.data.get('session_duration_seconds', 0)
                if session_time > 0:
                    window.metrics['session_times'].append(session_time)
            
        except Exception as e:
            logger.error(f"❌ Failed to add event to window {window.window_id}: {e}")
    
    def _get_tumbling_window_start(self, timestamp: datetime, duration_seconds: int) -> datetime:
        """Get start time for tumbling window"""
        # Align to window boundaries
        total_seconds = timestamp.timestamp()
        window_number = int(total_seconds // duration_seconds)
        return datetime.fromtimestamp(window_number * duration_seconds)
    
    def _get_sliding_window_starts(self, timestamp: datetime, duration_seconds: int,
                                 slide_interval_seconds: int) -> List[datetime]:
        """Get start times for sliding windows that contain this timestamp"""
        starts = []
        current_time = timestamp.timestamp()
        
        # Find the latest window start that's before this timestamp
        latest_start = int((current_time - duration_seconds) // slide_interval_seconds) * slide_interval_seconds
        
        # Generate window starts
        for i in range(duration_seconds // slide_interval_seconds + 1):
            start_time = latest_start + (i * slide_interval_seconds)
            start_datetime = datetime.fromtimestamp(start_time)
            
            # Check if timestamp falls within this window
            if start_datetime <= timestamp <= start_datetime + timedelta(seconds=duration_seconds):
                starts.append(start_datetime)
        
        return starts
    
    async def _cleanup_expired_windows(self, current_time: datetime):
        """Clean up expired windows"""
        try:
            expired_windows = []
            
            for window_id, window in self.active_windows.items():
                if current_time > window.end_time and not window.is_closed:
                    # Close the window
                    window.is_closed = True
                    window.closed_at = current_time
                    
                    # Calculate final metrics
                    await self._finalize_window_metrics(window)
                    
                    # Move to completed windows
                    self.completed_windows.append(window)
                    expired_windows.append(window_id)
            
            # Remove expired windows from active windows
            for window_id in expired_windows:
                del self.active_windows[window_id]
            
        except Exception as e:
            logger.error(f"❌ Window cleanup failed: {e}")
    
    async def _finalize_window_metrics(self, window: StreamWindow):
        """Calculate final metrics for a closed window"""
        try:
            # Calculate derived metrics
            if window.metrics.get('total_attempts', 0) > 0:
                window.metrics['success_rate'] = window.metrics['success_attempts'] / window.metrics['total_attempts']
            
            if window.metrics.get('session_times'):
                session_times = window.metrics['session_times']
                window.metrics['average_session_time'] = np.mean(session_times)
                window.metrics['median_session_time'] = np.median(session_times)
            
            window.metrics['active_users'] = len(window.unique_users)
            window.metrics['events_per_user'] = window.event_count / max(1, len(window.unique_users))
            
        except Exception as e:
            logger.error(f"❌ Window finalization failed: {e}")
    
    def get_current_metrics(self, metric_name: str) -> Dict[str, Any]:
        """Get current metrics for a specific metric"""
        try:
            # Find the most recent completed window for this metric
            matching_windows = [w for w in self.completed_windows 
                              if w.window_id.startswith(f"{metric_name}_")]
            
            if not matching_windows:
                return {}
            
            latest_window = max(matching_windows, key=lambda w: w.end_time)
            
            return {
                'window_id': latest_window.window_id,
                'end_time': latest_window.end_time,
                'metrics': latest_window.metrics,
                'event_count': latest_window.event_count,
                'unique_users': len(latest_window.unique_users)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get current metrics: {e}")
            return {}

class RealTimeAlertSystem:
    """Real-time alert system for streaming analytics"""
    
    def __init__(self):
        self.alert_rules = {}
        self.active_alerts = {}
        self.alert_history = deque(maxlen=10000)
        self.notification_handlers = []
    
    def add_alert_rule(self, rule_id: str, metric_name: str, condition: Callable,
                      alert_type: str, severity: str, message_template: str):
        """Add alert rule"""
        self.alert_rules[rule_id] = {
            'metric_name': metric_name,
            'condition': condition,
            'alert_type': alert_type,
            'severity': severity,
            'message_template': message_template,
            'last_triggered': None,
            'cooldown_seconds': 300  # 5 minutes default cooldown
        }
    
    def add_notification_handler(self, handler: Callable):
        """Add notification handler"""
        self.notification_handlers.append(handler)
    
    async def check_alerts(self, window: StreamWindow):
        """Check alert conditions for a completed window"""
        try:
            current_time = window.closed_at or datetime.now()
            
            for rule_id, rule in self.alert_rules.items():
                try:
                    # Check if we're in cooldown period
                    if rule['last_triggered']:
                        time_since_last = (current_time - rule['last_triggered']).total_seconds()
                        if time_since_last < rule['cooldown_seconds']:
                            continue
                    
                    # Check if condition is met
                    if rule['condition'](window.metrics):
                        await self._trigger_alert(rule_id, rule, window, current_time)
                        
                except Exception as e:
                    logger.error(f"❌ Alert rule {rule_id} evaluation failed: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Alert checking failed: {e}")
    
    async def _trigger_alert(self, rule_id: str, rule: Dict[str, Any], 
                           window: StreamWindow, current_time: datetime):
        """Trigger an alert"""
        try:
            alert_id = str(uuid.uuid4())
            
            # Create alert message
            message = rule['message_template'].format(
                window_id=window.window_id,
                metrics=window.metrics,
                unique_users=len(window.unique_users),
                event_count=window.event_count
            )
            
            alert = RealTimeAlert(
                alert_id=alert_id,
                alert_type=rule['alert_type'],
                severity=rule['severity'],
                message=message,
                triggered_at=current_time,
                affected_users=list(window.unique_users),
                affected_concepts=[],  # Would be extracted from context
                metric_values=window.metrics.copy()
            )
            
            # Store alert
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            # Update rule
            rule['last_triggered'] = current_time
            
            # Send notifications
            await self._send_notifications(alert)
            
            logger.info(f"🚨 Alert triggered: {rule['alert_type']} - {message}")
            
        except Exception as e:
            logger.error(f"❌ Alert triggering failed: {e}")
    
    async def _send_notifications(self, alert: RealTimeAlert):
        """Send alert notifications"""
        try:
            for handler in self.notification_handlers:
                try:
                    await handler(alert)
                except Exception as e:
                    logger.error(f"❌ Notification handler failed: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Notification sending failed: {e}")

class StreamingAnalyticsEngine:
    """Main streaming analytics engine"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/1"):
        self.redis_url = redis_url
        self.redis_client = None
        
        # Core components
        self.event_processor = EventProcessor()
        self.window_manager = WindowManager()
        self.alert_system = RealTimeAlertSystem()
        
        # Event streaming
        self.event_queue = Queue(maxsize=10000)
        self.processing_tasks = []
        self.is_running = False
        
        # Metrics storage
        self.live_metrics = {}
        self.metric_history = defaultdict(lambda: deque(maxlen=1000))
        
        # Performance monitoring
        self.processing_stats = {
            'events_received': 0,
            'events_processed': 0,
            'processing_errors': 0,
            'queue_size': 0,
            'processing_latency_ms': deque(maxlen=1000)
        }
    
    async def initialize(self):
        """Initialize streaming analytics engine"""
        try:
            logger.info("🚀 Initializing Streaming Analytics Engine")
            
            # Initialize Redis connection
            self.redis_client = aioredis.from_url(self.redis_url)
            
            # Configure default windows
            await self._configure_default_windows()
            
            # Register default event handlers
            await self._register_default_handlers()
            
            # Configure default alerts
            await self._configure_default_alerts()
            
            # Start processing tasks
            await self._start_processing_tasks()
            
            logger.info("✅ Streaming Analytics Engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Streaming Analytics Engine: {e}")
            return False
    
    async def _configure_default_windows(self):
        """Configure default time windows"""
        try:
            # 1-minute tumbling windows for real-time metrics
            self.window_manager.configure_window(
                "real_time_engagement", WindowType.TUMBLING, 60
            )
            
            # 5-minute sliding windows with 1-minute slides
            self.window_manager.configure_window(
                "rolling_performance", WindowType.SLIDING, 300, 60
            )
            
            # 15-minute tumbling windows for aggregated metrics
            self.window_manager.configure_window(
                "aggregated_metrics", WindowType.TUMBLING, 900
            )
            
            logger.info("⏰ Configured default time windows")
            
        except Exception as e:
            logger.error(f"❌ Window configuration failed: {e}")
    
    async def _register_default_handlers(self):
        """Register default event handlers"""
        try:
            # Register handlers for different event types
            self.event_processor.register_handler(
                EventType.PROBLEM_ATTEMPT, self._handle_problem_attempt
            )
            self.event_processor.register_handler(
                EventType.HELP_REQUEST, self._handle_help_request
            )
            self.event_processor.register_handler(
                EventType.INTERACTION, self._handle_interaction
            )
            self.event_processor.register_handler(
                EventType.LOGIN, self._handle_user_login
            )
            
            logger.info("🔧 Registered default event handlers")
            
        except Exception as e:
            logger.error(f"❌ Handler registration failed: {e}")
    
    async def _configure_default_alerts(self):
        """Configure default alert rules"""
        try:
            # Low success rate alert
            self.alert_system.add_alert_rule(
                "low_success_rate",
                "rolling_performance",
                lambda metrics: metrics.get('success_rate', 1.0) < 0.3,
                "performance_degradation",
                "high",
                "Low success rate detected: {metrics[success_rate]:.1%} in window {window_id}"
            )
            
            # High help request rate alert
            self.alert_system.add_alert_rule(
                "high_help_requests",
                "real_time_engagement",
                lambda metrics: metrics.get('help_requests', 0) > 100,
                "high_demand",
                "medium",
                "High help request volume: {metrics[help_requests]} requests in 1 minute"
            )
            
            # Low engagement alert
            self.alert_system.add_alert_rule(
                "low_engagement",
                "aggregated_metrics",
                lambda metrics: metrics.get('average_session_time', 1000) < 60,
                "engagement_drop",
                "medium",
                "Low average session time: {metrics[average_session_time]:.1f} seconds"
            )
            
            # Add notification handler
            self.alert_system.add_notification_handler(self._handle_alert_notification)
            
            logger.info("⚠️ Configured default alert rules")
            
        except Exception as e:
            logger.error(f"❌ Alert configuration failed: {e}")
    
    async def _start_processing_tasks(self):
        """Start background processing tasks"""
        try:
            self.is_running = True
            
            # Start event processing workers
            for i in range(3):  # 3 worker tasks
                task = asyncio.create_task(self._event_processing_worker(f"worker_{i}"))
                self.processing_tasks.append(task)
            
            # Start metrics updater
            metrics_task = asyncio.create_task(self._metrics_updater())
            self.processing_tasks.append(metrics_task)
            
            # Start statistics collector
            stats_task = asyncio.create_task(self._statistics_collector())
            self.processing_tasks.append(stats_task)
            
            logger.info("🔄 Started processing tasks")
            
        except Exception as e:
            logger.error(f"❌ Failed to start processing tasks: {e}")
    
    async def ingest_event(self, event: StreamEvent) -> bool:
        """Ingest a new event into the stream"""
        try:
            # Update statistics
            self.processing_stats['events_received'] += 1
            
            # Add to queue
            if self.event_queue.full():
                logger.warning("⚠️ Event queue is full, dropping oldest event")
                try:
                    self.event_queue.get_nowait()
                except:
                    pass
            
            await self.event_queue.put(event)
            self.processing_stats['queue_size'] = self.event_queue.qsize()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Event ingestion failed: {e}")
            self.processing_stats['processing_errors'] += 1
            return False
    
    async def _event_processing_worker(self, worker_id: str):
        """Event processing worker"""
        try:
            logger.info(f"🔄 Started event processing worker: {worker_id}")
            
            while self.is_running:
                try:
                    # Get event from queue with timeout
                    event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                    
                    processing_start = time.time()
                    
                    # Process event
                    await self.event_processor.process_event(event)
                    
                    # Process for windows
                    await self.window_manager.process_event_for_windows(event)
                    
                    # Update statistics
                    processing_time = (time.time() - processing_start) * 1000
                    self.processing_stats['processing_latency_ms'].append(processing_time)
                    self.processing_stats['events_processed'] += 1
                    self.processing_stats['queue_size'] = self.event_queue.qsize()
                    
                    # Mark task as done
                    self.event_queue.task_done()
                    
                except asyncio.TimeoutError:
                    continue  # No event available, continue loop
                except Exception as e:
                    logger.error(f"❌ Worker {worker_id} processing error: {e}")
                    self.processing_stats['processing_errors'] += 1
                    
        except Exception as e:
            logger.error(f"❌ Worker {worker_id} failed: {e}")
    
    async def _metrics_updater(self):
        """Update live metrics from completed windows"""
        try:
            while self.is_running:
                try:
                    # Process completed windows
                    for window in list(self.window_manager.completed_windows):
                        if not hasattr(window, '_processed_for_alerts'):
                            # Check alerts
                            await self.alert_system.check_alerts(window)
                            
                            # Update live metrics
                            await self._update_live_metrics(window)
                            
                            # Mark as processed
                            window._processed_for_alerts = True
                    
                    await asyncio.sleep(5)  # Check every 5 seconds
                    
                except Exception as e:
                    logger.error(f"❌ Metrics updater error: {e}")
                    await asyncio.sleep(10)
                    
        except Exception as e:
            logger.error(f"❌ Metrics updater failed: {e}")
    
    async def _update_live_metrics(self, window: StreamWindow):
        """Update live metrics from window data"""
        try:
            # Extract metric type from window ID
            metric_type = window.window_id.split('_')[0]
            
            # Update live metrics
            self.live_metrics[metric_type] = {
                'timestamp': window.end_time,
                'metrics': window.metrics.copy(),
                'event_count': window.event_count,
                'unique_users': len(window.unique_users),
                'window_duration': window.duration_seconds
            }
            
            # Store in history
            self.metric_history[metric_type].append(self.live_metrics[metric_type])
            
            # Store in Redis for persistence
            if self.redis_client:
                await self.redis_client.setex(
                    f"live_metrics:{metric_type}",
                    600,  # 10 minutes TTL
                    json.dumps(self.live_metrics[metric_type], default=str)
                )
                
        except Exception as e:
            logger.error(f"❌ Live metrics update failed: {e}")
    
    async def _statistics_collector(self):
        """Collect and log system statistics"""
        try:
            while self.is_running:
                try:
                    # Log processing statistics
                    stats = self.get_processing_statistics()
                    
                    if stats['events_processed'] > 0:
                        logger.info(
                            f"📊 Processing Stats - "
                            f"Events: {stats['events_processed']}, "
                            f"Queue: {stats['queue_size']}, "
                            f"Latency: {stats['avg_processing_latency_ms']:.1f}ms, "
                            f"Throughput: {stats['events_per_second']:.1f}/sec"
                        )
                    
                    await asyncio.sleep(30)  # Log every 30 seconds
                    
                except Exception as e:
                    logger.error(f"❌ Statistics collector error: {e}")
                    await asyncio.sleep(60)
                    
        except Exception as e:
            logger.error(f"❌ Statistics collector failed: {e}")
    
    # Event Handlers
    async def _handle_problem_attempt(self, event: StreamEvent) -> Dict[str, Any]:
        """Handle problem attempt events"""
        try:
            success = event.data.get('success', False)
            concept = event.data.get('concept', 'unknown')
            difficulty = event.data.get('difficulty', 0.5)
            
            # Immediate feedback for adaptive learning
            if not success and difficulty > 0.8:
                # Student struggling with difficult problem
                adaptive_action = {
                    'action': 'reduce_difficulty',
                    'user_id': event.user_id,
                    'concept': concept,
                    'reason': 'struggling_with_difficult_content'
                }
                
                # Could trigger immediate intervention
                await self._trigger_adaptive_action(adaptive_action)
            
            return {
                'processed': True,
                'success': success,
                'concept': concept,
                'adaptive_action_triggered': not success and difficulty > 0.8
            }
            
        except Exception as e:
            logger.error(f"❌ Problem attempt handler failed: {e}")
            return {'error': str(e)}
    
    async def _handle_help_request(self, event: StreamEvent) -> Dict[str, Any]:
        """Handle help request events"""
        try:
            help_type = event.data.get('help_type', 'general')
            concept = event.data.get('concept', 'unknown')
            
            # Track help patterns for concept difficulty analysis
            result = {
                'processed': True,
                'help_type': help_type,
                'concept': concept
            }
            
            # Check for repeated help requests (indicating confusion)
            recent_help_count = await self._count_recent_help_requests(event.user_id, concept)
            if recent_help_count > 3:
                # Multiple help requests suggest the student needs intervention
                intervention = {
                    'action': 'suggest_alternative_explanation',
                    'user_id': event.user_id,
                    'concept': concept,
                    'reason': 'repeated_help_requests'
                }
                
                await self._trigger_adaptive_action(intervention)
                result['intervention_triggered'] = True
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Help request handler failed: {e}")
            return {'error': str(e)}
    
    async def _handle_interaction(self, event: StreamEvent) -> Dict[str, Any]:
        """Handle general interaction events"""
        try:
            interaction_type = event.data.get('type', 'general')
            duration = event.data.get('duration_seconds', 0)
            
            return {
                'processed': True,
                'interaction_type': interaction_type,
                'duration': duration
            }
            
        except Exception as e:
            logger.error(f"❌ Interaction handler failed: {e}")
            return {'error': str(e)}
    
    async def _handle_user_login(self, event: StreamEvent) -> Dict[str, Any]:
        """Handle user login events"""
        try:
            # Update active user tracking
            await self._update_active_users(event.user_id, event.timestamp)
            
            return {
                'processed': True,
                'user_id': event.user_id,
                'login_time': event.timestamp
            }
            
        except Exception as e:
            logger.error(f"❌ User login handler failed: {e}")
            return {'error': str(e)}
    
    async def _trigger_adaptive_action(self, action: Dict[str, Any]):
        """Trigger adaptive learning action"""
        try:
            # In a real system, this would trigger actions in the learning system
            logger.info(f"🎯 Adaptive action triggered: {action['action']} for {action['user_id']}")
            
            # Store action for tracking
            if self.redis_client:
                await self.redis_client.lpush(
                    f"adaptive_actions:{action['user_id']}", 
                    json.dumps(action, default=str)
                )
                
        except Exception as e:
            logger.error(f"❌ Adaptive action trigger failed: {e}")
    
    async def _count_recent_help_requests(self, user_id: str, concept: str, 
                                        minutes: int = 10) -> int:
        """Count recent help requests for a user/concept"""
        try:
            if not self.redis_client:
                return 0
            
            # Simple implementation using Redis
            key = f"help_requests:{user_id}:{concept}"
            count = await self.redis_client.get(key)
            
            return int(count) if count else 0
            
        except Exception as e:
            logger.error(f"❌ Help request counting failed: {e}")
            return 0
    
    async def _update_active_users(self, user_id: str, timestamp: datetime):
        """Update active users tracking"""
        try:
            if self.redis_client:
                # Add to active users set with expiration
                await self.redis_client.setex(f"active_user:{user_id}", 1800, "1")  # 30 minutes
                
        except Exception as e:
            logger.error(f"❌ Active users update failed: {e}")
    
    async def _handle_alert_notification(self, alert: RealTimeAlert):
        """Handle alert notification"""
        try:
            # Log the alert
            logger.warning(f"🚨 ALERT: {alert.alert_type} - {alert.message}")
            
            # Store in Redis for dashboard
            if self.redis_client:
                await self.redis_client.lpush(
                    "real_time_alerts",
                    json.dumps(asdict(alert), default=str)
                )
                
                # Keep only recent alerts
                await self.redis_client.ltrim("real_time_alerts", 0, 999)
                
        except Exception as e:
            logger.error(f"❌ Alert notification handling failed: {e}")
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        try:
            latencies = list(self.processing_stats['processing_latency_ms'])
            
            stats = {
                'events_received': self.processing_stats['events_received'],
                'events_processed': self.processing_stats['events_processed'],
                'processing_errors': self.processing_stats['processing_errors'],
                'queue_size': self.processing_stats['queue_size'],
                'error_rate': self.processing_stats['processing_errors'] / max(1, self.processing_stats['events_received']),
                'avg_processing_latency_ms': np.mean(latencies) if latencies else 0,
                'p95_processing_latency_ms': np.percentile(latencies, 95) if len(latencies) > 10 else 0,
                'events_per_second': 0
            }
            
            # Calculate throughput
            if latencies:
                total_time_seconds = sum(latencies) / 1000
                stats['events_per_second'] = len(latencies) / max(1, total_time_seconds)
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Statistics calculation failed: {e}")
            return {}
    
    async def get_live_metrics(self) -> Dict[str, Any]:
        """Get current live metrics"""
        try:
            return {
                'live_metrics': self.live_metrics.copy(),
                'processing_stats': self.get_processing_statistics(),
                'active_windows': len(self.window_manager.active_windows),
                'completed_windows': len(self.window_manager.completed_windows),
                'active_alerts': len(self.alert_system.active_alerts),
                'alert_history_size': len(self.alert_system.alert_history)
            }
            
        except Exception as e:
            logger.error(f"❌ Live metrics retrieval failed: {e}")
            return {}
    
    async def cleanup(self):
        """Cleanup streaming analytics engine"""
        try:
            self.is_running = False
            
            # Cancel processing tasks
            for task in self.processing_tasks:
                task.cancel()
            
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("✅ Streaming analytics engine cleaned up")
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")

# Testing function
async def test_streaming_analytics():
    """Test streaming analytics engine"""
    try:
        logger.info("🧪 Testing Streaming Analytics Engine")
        
        # Initialize engine
        engine = StreamingAnalyticsEngine()
        await engine.initialize()
        
        # Simulate events
        logger.info("📡 Simulating events...")
        
        # Create test events
        for i in range(100):
            # Problem attempt events
            event = StreamEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.PROBLEM_ATTEMPT,
                timestamp=datetime.now() - timedelta(seconds=100-i),
                user_id=f"user_{i % 20}",  # 20 different users
                session_id=f"session_{i % 50}",
                data={
                    'success': i % 3 != 0,  # 66% success rate
                    'concept': ['mechanics', 'energy', 'waves'][i % 3],
                    'difficulty': 0.3 + (i % 7) * 0.1,
                    'response_time_ms': 1000 + i * 10
                }
            )
            
            await engine.ingest_event(event)
            
            # Help request events (less frequent)
            if i % 5 == 0:
                help_event = StreamEvent(
                    event_id=str(uuid.uuid4()),
                    event_type=EventType.HELP_REQUEST,
                    timestamp=datetime.now() - timedelta(seconds=100-i),
                    user_id=f"user_{i % 20}",
                    session_id=f"session_{i % 50}",
                    data={
                        'help_type': 'explanation',
                        'concept': ['mechanics', 'energy', 'waves'][i % 3]
                    }
                )
                
                await engine.ingest_event(help_event)
        
        # Wait for processing
        await asyncio.sleep(5)
        
        # Check results
        live_metrics = await engine.get_live_metrics()
        logger.info(f"📊 Live Metrics: {json.dumps(live_metrics, indent=2, default=str)}")
        
        processing_stats = engine.get_processing_statistics()
        logger.info(f"⚡ Processing: {processing_stats['events_per_second']:.1f} events/sec")
        logger.info(f"🎯 Success Rate in Windows: Check completed windows for success rates")
        
        # Cleanup
        await engine.cleanup()
        
        logger.info("✅ Streaming Analytics Engine test completed")
        
    except Exception as e:
        logger.error(f"❌ Streaming Analytics test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_streaming_analytics())