#!/usr/bin/env python3
"""
Real-time Analytics Processing Pipeline for Physics Assistant
Event-driven analytics system that processes student interactions in real-time,
updates learning models, triggers adaptive interventions, and provides
live insights for educators and students.
"""

import asyncio
import json
import logging
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
import weakref
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EventType(Enum):
    """Types of analytics events"""
    INTERACTION_START = "interaction_start"
    INTERACTION_SUCCESS = "interaction_success"
    INTERACTION_FAILURE = "interaction_failure"
    CONCEPT_MASTERY_UPDATE = "concept_mastery_update"
    LEARNING_PATH_PROGRESS = "learning_path_progress"
    PERFORMANCE_THRESHOLD = "performance_threshold"
    ENGAGEMENT_CHANGE = "engagement_change"
    ERROR_PATTERN_DETECTED = "error_pattern_detected"
    INTERVENTION_NEEDED = "intervention_needed"

class TriggerCondition(Enum):
    """Conditions for triggering analytics events"""
    IMMEDIATE = "immediate"
    THRESHOLD_REACHED = "threshold_reached"
    PATTERN_DETECTED = "pattern_detected"
    TIME_BASED = "time_based"
    ACCUMULATIVE = "accumulative"

@dataclass
class AnalyticsEvent:
    """Real-time analytics event"""
    event_id: str
    event_type: EventType
    user_id: str
    timestamp: datetime
    data: Dict[str, Any]
    context: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1=low, 2=medium, 3=high, 4=critical
    processed: bool = False

@dataclass
class RealTimeMetric:
    """Real-time metric tracking"""
    metric_name: str
    current_value: float
    previous_value: float
    timestamp: datetime
    trend: str  # 'increasing', 'decreasing', 'stable'
    threshold_alerts: List[str] = field(default_factory=list)

@dataclass
class StreamingUpdate:
    """Update to be streamed to dashboards"""
    update_id: str
    user_id: Optional[str]
    metric_type: str
    data: Dict[str, Any]
    timestamp: datetime
    dashboard_targets: List[str] = field(default_factory=list)

@dataclass
class InterventionTrigger:
    """Trigger for adaptive intervention"""
    trigger_id: str
    user_id: str
    trigger_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    recommended_actions: List[str]
    auto_execute: bool = False
    timestamp: datetime = field(default_factory=datetime.now)

class EventProcessor:
    """Base class for event processors"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_active = True
        self.processed_count = 0
        
    async def process(self, event: AnalyticsEvent) -> List[StreamingUpdate]:
        """Process an analytics event"""
        raise NotImplementedError
    
    def can_process(self, event: AnalyticsEvent) -> bool:
        """Check if this processor can handle the event"""
        return True

class InteractionProcessor(EventProcessor):
    """Processor for student interaction events"""
    
    def __init__(self):
        super().__init__("interaction_processor")
        self.interaction_cache = defaultdict(deque)
        self.success_rates = defaultdict(float)
        
    async def process(self, event: AnalyticsEvent) -> List[StreamingUpdate]:
        updates = []
        user_id = event.user_id
        
        try:
            # Cache interaction
            self.interaction_cache[user_id].append(event)
            
            # Keep only recent interactions (sliding window)
            if len(self.interaction_cache[user_id]) > 50:
                self.interaction_cache[user_id].popleft()
            
            # Calculate real-time success rate
            recent_interactions = list(self.interaction_cache[user_id])[-10:]
            if recent_interactions:
                successes = sum(1 for e in recent_interactions 
                              if e.event_type == EventType.INTERACTION_SUCCESS)
                success_rate = successes / len(recent_interactions)
                
                # Update success rate
                previous_rate = self.success_rates[user_id]
                self.success_rates[user_id] = success_rate
                
                # Create streaming update
                update = StreamingUpdate(
                    update_id=f"success_rate_{user_id}_{event.timestamp.timestamp()}",
                    user_id=user_id,
                    metric_type="success_rate",
                    data={
                        "current_rate": success_rate,
                        "previous_rate": previous_rate,
                        "trend": "improving" if success_rate > previous_rate else "declining" if success_rate < previous_rate else "stable",
                        "sample_size": len(recent_interactions)
                    },
                    timestamp=event.timestamp,
                    dashboard_targets=["student_progress", "educator_overview"]
                )
                updates.append(update)
            
            self.processed_count += 1
            
        except Exception as e:
            logger.error(f"❌ Error processing interaction event: {e}")
        
        return updates
    
    def can_process(self, event: AnalyticsEvent) -> bool:
        return event.event_type in [EventType.INTERACTION_SUCCESS, EventType.INTERACTION_FAILURE]

class MasteryProcessor(EventProcessor):
    """Processor for concept mastery events"""
    
    def __init__(self):
        super().__init__("mastery_processor")
        self.mastery_levels = defaultdict(dict)
        self.mastery_thresholds = {"beginner": 0.6, "intermediate": 0.75, "advanced": 0.9}
        
    async def process(self, event: AnalyticsEvent) -> List[StreamingUpdate]:
        updates = []
        user_id = event.user_id
        
        try:
            if "concept" in event.data and "mastery_level" in event.data:
                concept = event.data["concept"]
                new_mastery = event.data["mastery_level"]
                
                # Update mastery level
                old_mastery = self.mastery_levels[user_id].get(concept, 0.0)
                self.mastery_levels[user_id][concept] = new_mastery
                
                # Check for threshold crossings
                threshold_alerts = []
                for level, threshold in self.mastery_thresholds.items():
                    if old_mastery < threshold <= new_mastery:
                        threshold_alerts.append(f"Achieved {level} mastery in {concept}")
                    elif old_mastery >= threshold > new_mastery:
                        threshold_alerts.append(f"Dropped below {level} mastery in {concept}")
                
                # Create streaming update
                update = StreamingUpdate(
                    update_id=f"mastery_{user_id}_{concept}_{event.timestamp.timestamp()}",
                    user_id=user_id,
                    metric_type="concept_mastery",
                    data={
                        "concept": concept,
                        "mastery_level": new_mastery,
                        "previous_level": old_mastery,
                        "change": new_mastery - old_mastery,
                        "threshold_alerts": threshold_alerts
                    },
                    timestamp=event.timestamp,
                    dashboard_targets=["student_progress", "concept_overview"]
                )
                updates.append(update)
            
            self.processed_count += 1
            
        except Exception as e:
            logger.error(f"❌ Error processing mastery event: {e}")
        
        return updates
    
    def can_process(self, event: AnalyticsEvent) -> bool:
        return event.event_type == EventType.CONCEPT_MASTERY_UPDATE

class EngagementProcessor(EventProcessor):
    """Processor for engagement-related events"""
    
    def __init__(self):
        super().__init__("engagement_processor")
        self.session_data = defaultdict(dict)
        self.engagement_scores = defaultdict(float)
        
    async def process(self, event: AnalyticsEvent) -> List[StreamingUpdate]:
        updates = []
        user_id = event.user_id
        
        try:
            # Track session activity
            current_time = event.timestamp
            session_key = f"{user_id}_{current_time.date()}"
            
            if session_key not in self.session_data:
                self.session_data[session_key] = {
                    "start_time": current_time,
                    "last_activity": current_time,
                    "interaction_count": 0,
                    "success_count": 0
                }
            
            session = self.session_data[session_key]
            session["last_activity"] = current_time
            session["interaction_count"] += 1
            
            if event.event_type == EventType.INTERACTION_SUCCESS:
                session["success_count"] += 1
            
            # Calculate engagement score
            session_duration = (current_time - session["start_time"]).total_seconds() / 60  # minutes
            interaction_rate = session["interaction_count"] / max(session_duration, 1)
            success_rate = session["success_count"] / session["interaction_count"]
            
            # Simple engagement score calculation
            engagement_score = min(1.0, (interaction_rate * 0.1) + success_rate) * 0.5 + 0.5
            
            previous_score = self.engagement_scores[user_id]
            self.engagement_scores[user_id] = engagement_score
            
            # Create streaming update
            update = StreamingUpdate(
                update_id=f"engagement_{user_id}_{event.timestamp.timestamp()}",
                user_id=user_id,
                metric_type="engagement",
                data={
                    "engagement_score": engagement_score,
                    "previous_score": previous_score,
                    "session_duration": session_duration,
                    "interaction_rate": interaction_rate,
                    "session_success_rate": success_rate
                },
                timestamp=event.timestamp,
                dashboard_targets=["student_engagement", "educator_overview"]
            )
            updates.append(update)
            
            self.processed_count += 1
            
        except Exception as e:
            logger.error(f"❌ Error processing engagement event: {e}")
        
        return updates

class ErrorPatternProcessor(EventProcessor):
    """Processor for detecting error patterns in real-time"""
    
    def __init__(self):
        super().__init__("error_pattern_processor")
        self.error_sequences = defaultdict(deque)
        self.pattern_thresholds = {"repetitive": 3, "alternating": 4, "escalating": 2}
        
    async def process(self, event: AnalyticsEvent) -> List[StreamingUpdate]:
        updates = []
        user_id = event.user_id
        
        try:
            if event.event_type == EventType.INTERACTION_FAILURE:
                # Track error sequence
                concept = event.data.get("concept", "unknown")
                error_type = event.data.get("error_type", "general")
                
                self.error_sequences[user_id].append({
                    "concept": concept,
                    "error_type": error_type,
                    "timestamp": event.timestamp
                })
                
                # Keep only recent errors (last 10)
                if len(self.error_sequences[user_id]) > 10:
                    self.error_sequences[user_id].popleft()
                
                # Detect patterns
                patterns = self._detect_error_patterns(list(self.error_sequences[user_id]))
                
                if patterns:
                    update = StreamingUpdate(
                        update_id=f"error_pattern_{user_id}_{event.timestamp.timestamp()}",
                        user_id=user_id,
                        metric_type="error_pattern",
                        data={
                            "detected_patterns": patterns,
                            "recent_errors": len(self.error_sequences[user_id]),
                            "severity": self._calculate_pattern_severity(patterns)
                        },
                        timestamp=event.timestamp,
                        dashboard_targets=["student_support", "educator_alerts"]
                    )
                    updates.append(update)
            
            self.processed_count += 1
            
        except Exception as e:
            logger.error(f"❌ Error processing error pattern event: {e}")
        
        return updates
    
    def _detect_error_patterns(self, error_sequence: List[Dict]) -> List[str]:
        """Detect patterns in error sequence"""
        patterns = []
        
        if len(error_sequence) < 2:
            return patterns
        
        # Repetitive errors (same concept/error type)
        recent_errors = error_sequence[-3:]
        if len(recent_errors) >= 2:
            if all(e["concept"] == recent_errors[0]["concept"] for e in recent_errors):
                patterns.append("repetitive_concept_errors")
            if all(e["error_type"] == recent_errors[0]["error_type"] for e in recent_errors):
                patterns.append("repetitive_error_type")
        
        # Alternating pattern
        if len(error_sequence) >= 4:
            concepts = [e["concept"] for e in error_sequence[-4:]]
            if concepts[0] == concepts[2] and concepts[1] == concepts[3] and concepts[0] != concepts[1]:
                patterns.append("alternating_concept_errors")
        
        # Escalating difficulty (concept progression errors)
        if len(error_sequence) >= 3:
            # This would require concept difficulty mapping
            patterns.append("potential_escalating_difficulty")
        
        return patterns
    
    def _calculate_pattern_severity(self, patterns: List[str]) -> str:
        """Calculate severity of detected patterns"""
        if "repetitive_error_type" in patterns:
            return "high"
        elif "repetitive_concept_errors" in patterns:
            return "medium"
        elif "alternating_concept_errors" in patterns:
            return "medium"
        else:
            return "low"
    
    def can_process(self, event: AnalyticsEvent) -> bool:
        return event.event_type in [EventType.INTERACTION_FAILURE, EventType.ERROR_PATTERN_DETECTED]

class InterventionEngine:
    """Engine for triggering adaptive interventions"""
    
    def __init__(self):
        self.intervention_rules = []
        self.active_interventions = defaultdict(list)
        self.intervention_history = defaultdict(list)
        self._setup_default_rules()
        
    def _setup_default_rules(self):
        """Setup default intervention rules"""
        self.intervention_rules = [
            {
                "name": "low_success_rate",
                "condition": lambda data: data.get("success_rate", 1.0) < 0.4,
                "severity": "medium",
                "actions": ["provide_hints", "reduce_difficulty", "offer_review_materials"],
                "cooldown_minutes": 30
            },
            {
                "name": "repetitive_errors",
                "condition": lambda data: "repetitive_error_type" in data.get("detected_patterns", []),
                "severity": "high",
                "actions": ["targeted_remediation", "concept_review", "peer_assistance"],
                "cooldown_minutes": 15
            },
            {
                "name": "low_engagement",
                "condition": lambda data: data.get("engagement_score", 1.0) < 0.3,
                "severity": "medium",
                "actions": ["gamification_boost", "break_suggestion", "motivational_message"],
                "cooldown_minutes": 45
            },
            {
                "name": "mastery_plateau",
                "condition": lambda data: data.get("change", 0) == 0 and data.get("mastery_level", 0) < 0.7,
                "severity": "low",
                "actions": ["alternative_approach", "additional_practice", "concept_explanation"],
                "cooldown_minutes": 60
            }
        ]
    
    async def evaluate_interventions(self, update: StreamingUpdate) -> List[InterventionTrigger]:
        """Evaluate if interventions should be triggered"""
        triggers = []
        
        try:
            if not update.user_id:
                return triggers
            
            user_id = update.user_id
            current_time = update.timestamp
            
            # Check each intervention rule
            for rule in self.intervention_rules:
                if rule["condition"](update.data):
                    # Check cooldown
                    if self._is_intervention_on_cooldown(user_id, rule["name"], current_time):
                        continue
                    
                    # Create intervention trigger
                    trigger = InterventionTrigger(
                        trigger_id=f"{rule['name']}_{user_id}_{current_time.timestamp()}",
                        user_id=user_id,
                        trigger_type=rule["name"],
                        severity=rule["severity"],
                        description=f"Intervention needed: {rule['name']}",
                        recommended_actions=rule["actions"],
                        auto_execute=rule["severity"] in ["high", "critical"]
                    )
                    triggers.append(trigger)
                    
                    # Record intervention
                    self.intervention_history[user_id].append({
                        "trigger_type": rule["name"],
                        "timestamp": current_time,
                        "severity": rule["severity"]
                    })
        
        except Exception as e:
            logger.error(f"❌ Error evaluating interventions: {e}")
        
        return triggers
    
    def _is_intervention_on_cooldown(self, user_id: str, intervention_type: str, current_time: datetime) -> bool:
        """Check if intervention is on cooldown"""
        history = self.intervention_history.get(user_id, [])
        
        # Find matching intervention rule
        rule = next((r for r in self.intervention_rules if r["name"] == intervention_type), None)
        if not rule:
            return False
        
        cooldown_minutes = rule["cooldown_minutes"]
        
        # Check recent history
        for record in history:
            if record["trigger_type"] == intervention_type:
                time_diff = (current_time - record["timestamp"]).total_seconds() / 60
                if time_diff < cooldown_minutes:
                    return True
        
        return False

class RealTimeAnalyticsEngine:
    """Main real-time analytics processing engine"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.event_queue = asyncio.Queue()
        self.processors = []
        self.intervention_engine = InterventionEngine()
        self.streaming_clients = defaultdict(list)  # Dashboard connections
        self.is_running = False
        self.metrics = defaultdict(lambda: defaultdict(float))
        
        # Performance tracking
        self.processing_stats = {
            "events_processed": 0,
            "processing_errors": 0,
            "avg_processing_time": 0.0,
            "updates_generated": 0
        }
        
        # Initialize processors
        self._initialize_processors()
        
    def _initialize_processors(self):
        """Initialize event processors"""
        self.processors = [
            InteractionProcessor(),
            MasteryProcessor(),
            EngagementProcessor(),
            ErrorPatternProcessor()
        ]
    
    async def start(self):
        """Start the real-time analytics engine"""
        try:
            logger.info("🚀 Starting Real-time Analytics Engine")
            self.is_running = True
            
            # Start processing tasks
            tasks = [
                asyncio.create_task(self._event_processing_loop()),
                asyncio.create_task(self._metrics_collection_loop()),
                asyncio.create_task(self._cleanup_loop())
            ]
            
            logger.info("✅ Real-time Analytics Engine started successfully")
            
            # Wait for tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"❌ Failed to start Real-time Analytics Engine: {e}")
            self.is_running = False
    
    async def stop(self):
        """Stop the real-time analytics engine"""
        logger.info("🔒 Stopping Real-time Analytics Engine")
        self.is_running = False
        
        # Clear queues and cleanup
        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        logger.info("✅ Real-time Analytics Engine stopped")
    
    async def submit_event(self, event: AnalyticsEvent):
        """Submit an event for processing"""
        try:
            await self.event_queue.put(event)
        except Exception as e:
            logger.error(f"❌ Failed to submit event: {e}")
    
    async def _event_processing_loop(self):
        """Main event processing loop"""
        while self.is_running:
            try:
                # Get event from queue with timeout
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                start_time = time.time()
                
                # Process event with all applicable processors
                all_updates = []
                for processor in self.processors:
                    if processor.can_process(event) and processor.is_active:
                        try:
                            updates = await processor.process(event)
                            all_updates.extend(updates)
                        except Exception as e:
                            logger.error(f"❌ Processor {processor.name} failed: {e}")
                            self.processing_stats["processing_errors"] += 1
                
                # Evaluate interventions
                intervention_triggers = []
                for update in all_updates:
                    triggers = await self.intervention_engine.evaluate_interventions(update)
                    intervention_triggers.extend(triggers)
                
                # Stream updates to dashboards
                await self._stream_updates(all_updates)
                
                # Process intervention triggers
                await self._process_intervention_triggers(intervention_triggers)
                
                # Update processing stats
                processing_time = time.time() - start_time
                self.processing_stats["events_processed"] += 1
                self.processing_stats["updates_generated"] += len(all_updates)
                
                # Update average processing time
                current_avg = self.processing_stats["avg_processing_time"]
                count = self.processing_stats["events_processed"]
                self.processing_stats["avg_processing_time"] = (current_avg * (count - 1) + processing_time) / count
                
                # Mark event as processed
                event.processed = True
                
            except asyncio.TimeoutError:
                # No events to process, continue
                continue
            except Exception as e:
                logger.error(f"❌ Error in event processing loop: {e}")
                self.processing_stats["processing_errors"] += 1
    
    async def _stream_updates(self, updates: List[StreamingUpdate]):
        """Stream updates to connected dashboards"""
        try:
            for update in updates:
                # Send to specific user dashboard
                if update.user_id and update.user_id in self.streaming_clients:
                    for client_callback in self.streaming_clients[update.user_id]:
                        try:
                            await client_callback(update)
                        except Exception as e:
                            logger.error(f"❌ Failed to stream to client: {e}")
                
                # Send to general dashboard targets
                for target in update.dashboard_targets:
                    if target in self.streaming_clients:
                        for client_callback in self.streaming_clients[target]:
                            try:
                                await client_callback(update)
                            except Exception as e:
                                logger.error(f"❌ Failed to stream to dashboard {target}: {e}")
        
        except Exception as e:
            logger.error(f"❌ Error streaming updates: {e}")
    
    async def _process_intervention_triggers(self, triggers: List[InterventionTrigger]):
        """Process intervention triggers"""
        try:
            for trigger in triggers:
                # Log intervention
                logger.info(f"🚨 Intervention triggered: {trigger.trigger_type} for user {trigger.user_id}")
                
                # Auto-execute high severity interventions
                if trigger.auto_execute:
                    await self._execute_intervention(trigger)
                
                # Store intervention trigger for manual review
                await self._store_intervention_trigger(trigger)
        
        except Exception as e:
            logger.error(f"❌ Error processing intervention triggers: {e}")
    
    async def _execute_intervention(self, trigger: InterventionTrigger):
        """Execute an intervention automatically"""
        try:
            # This would integrate with the main system to execute interventions
            # For now, just log the action
            logger.info(f"🤖 Auto-executing intervention: {trigger.recommended_actions}")
            
            # Example: Send notification, adjust difficulty, provide hints, etc.
            # Implementation would depend on the main system architecture
            
        except Exception as e:
            logger.error(f"❌ Failed to execute intervention: {e}")
    
    async def _store_intervention_trigger(self, trigger: InterventionTrigger):
        """Store intervention trigger in database"""
        try:
            if self.db_manager:
                # Store in database for later review
                async with self.db_manager.postgres.get_connection() as conn:
                    await conn.execute("""
                        INSERT INTO intervention_triggers 
                        (trigger_id, user_id, trigger_type, severity, description, 
                         recommended_actions, auto_executed, created_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """, 
                    trigger.trigger_id, trigger.user_id, trigger.trigger_type,
                    trigger.severity, trigger.description, 
                    json.dumps(trigger.recommended_actions), trigger.auto_execute,
                    trigger.timestamp)
        
        except Exception as e:
            logger.error(f"❌ Failed to store intervention trigger: {e}")
    
    async def _metrics_collection_loop(self):
        """Collect and update system metrics"""
        while self.is_running:
            try:
                await asyncio.sleep(10)  # Collect metrics every 10 seconds
                
                # Collect processor metrics
                for processor in self.processors:
                    self.metrics["processors"][processor.name] = processor.processed_count
                
                # Collect queue metrics
                self.metrics["system"]["queue_size"] = self.event_queue.qsize()
                self.metrics["system"]["connected_clients"] = sum(len(clients) for clients in self.streaming_clients.values())
                
                # Update processing stats
                for key, value in self.processing_stats.items():
                    self.metrics["performance"][key] = value
                
            except Exception as e:
                logger.error(f"❌ Error in metrics collection: {e}")
    
    async def _cleanup_loop(self):
        """Periodic cleanup of old data"""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
                current_time = datetime.now()
                cutoff_time = current_time - timedelta(hours=1)
                
                # Cleanup processor caches
                for processor in self.processors:
                    if hasattr(processor, 'interaction_cache'):
                        for user_id, cache in processor.interaction_cache.items():
                            # Remove old events
                            while cache and cache[0].timestamp < cutoff_time:
                                cache.popleft()
                
                # Cleanup intervention history
                for user_id, history in self.intervention_engine.intervention_history.items():
                    self.intervention_engine.intervention_history[user_id] = [
                        record for record in history 
                        if record["timestamp"] > cutoff_time
                    ]
                
            except Exception as e:
                logger.error(f"❌ Error in cleanup loop: {e}")
    
    def register_streaming_client(self, client_id: str, callback: Callable):
        """Register a client for streaming updates"""
        try:
            self.streaming_clients[client_id].append(callback)
            logger.info(f"📡 Registered streaming client: {client_id}")
        except Exception as e:
            logger.error(f"❌ Failed to register streaming client: {e}")
    
    def unregister_streaming_client(self, client_id: str, callback: Callable):
        """Unregister a streaming client"""
        try:
            if client_id in self.streaming_clients:
                self.streaming_clients[client_id] = [
                    cb for cb in self.streaming_clients[client_id] if cb != callback
                ]
                if not self.streaming_clients[client_id]:
                    del self.streaming_clients[client_id]
            logger.info(f"📡 Unregistered streaming client: {client_id}")
        except Exception as e:
            logger.error(f"❌ Failed to unregister streaming client: {e}")
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        return dict(self.metrics)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.processing_stats.copy()

# Utility functions for creating events
def create_interaction_event(user_id: str, agent_type: str, success: bool, 
                           execution_time: int = None, context: Dict = None) -> AnalyticsEvent:
    """Create an interaction analytics event"""
    event_type = EventType.INTERACTION_SUCCESS if success else EventType.INTERACTION_FAILURE
    
    return AnalyticsEvent(
        event_id=f"interaction_{user_id}_{datetime.now().timestamp()}",
        event_type=event_type,
        user_id=user_id,
        timestamp=datetime.now(),
        data={
            "agent_type": agent_type,
            "success": success,
            "execution_time_ms": execution_time,
            "concept": agent_type
        },
        context=context or {}
    )

def create_mastery_event(user_id: str, concept: str, mastery_level: float, 
                        confidence: float = None) -> AnalyticsEvent:
    """Create a concept mastery analytics event"""
    return AnalyticsEvent(
        event_id=f"mastery_{user_id}_{concept}_{datetime.now().timestamp()}",
        event_type=EventType.CONCEPT_MASTERY_UPDATE,
        user_id=user_id,
        timestamp=datetime.now(),
        data={
            "concept": concept,
            "mastery_level": mastery_level,
            "confidence_level": confidence
        }
    )

def create_error_pattern_event(user_id: str, error_type: str, concept: str, 
                              context: Dict = None) -> AnalyticsEvent:
    """Create an error pattern analytics event"""
    return AnalyticsEvent(
        event_id=f"error_{user_id}_{error_type}_{datetime.now().timestamp()}",
        event_type=EventType.ERROR_PATTERN_DETECTED,
        user_id=user_id,
        timestamp=datetime.now(),
        data={
            "error_type": error_type,
            "concept": concept
        },
        context=context or {},
        priority=2  # Medium priority
    )

# Example usage and testing
async def test_realtime_analytics():
    """Test function for real-time analytics"""
    try:
        logger.info("🧪 Testing Real-time Analytics Engine")
        
        # Initialize engine
        engine = RealTimeAnalyticsEngine()
        
        # Start engine in background
        engine_task = asyncio.create_task(engine.start())
        
        # Give it time to start
        await asyncio.sleep(1)
        
        # Create test events
        test_events = [
            create_interaction_event("test_user_1", "kinematics", True, 15000),
            create_interaction_event("test_user_1", "kinematics", False, 25000),
            create_mastery_event("test_user_1", "kinematics", 0.75),
            create_error_pattern_event("test_user_1", "unit_error", "kinematics")
        ]
        
        # Submit test events
        for event in test_events:
            await engine.submit_event(event)
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Check metrics
        metrics = engine.get_system_metrics()
        stats = engine.get_processing_stats()
        
        logger.info(f"✅ Processed {stats['events_processed']} events")
        logger.info(f"✅ Generated {stats['updates_generated']} updates")
        logger.info(f"✅ Processing errors: {stats['processing_errors']}")
        logger.info(f"✅ Average processing time: {stats['avg_processing_time']:.3f}s")
        
        # Stop engine
        await engine.stop()
        engine_task.cancel()
        
        logger.info("✅ Real-time Analytics test completed")
        
    except Exception as e:
        logger.error(f"❌ Real-time Analytics test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_realtime_analytics())