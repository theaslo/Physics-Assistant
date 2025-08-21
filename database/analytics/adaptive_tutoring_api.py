#!/usr/bin/env python3
"""
Phase 6.2 Adaptive Tutoring API Server
Real-time intelligent tutoring system with <200ms response time,
personalized learning paths, and physics-specific educational intelligence.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Import the enhanced intelligent tutoring engine
from intelligent_tutoring_engine import IntelligentTutoringEngine, LearningStyle, MasteryState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics for tutoring system
TUTORING_REQUESTS = Counter(
    'tutoring_requests_total',
    'Total tutoring system requests',
    ['endpoint', 'student_id']
)
ADAPTATION_TIME = Histogram(
    'adaptation_response_time_seconds',
    'Time taken for adaptive responses',
    ['operation_type']
)
ACTIVE_SESSIONS = Gauge(
    'active_tutoring_sessions',
    'Number of active tutoring sessions'
)
MASTERY_ACHIEVEMENTS = Counter(
    'mastery_achievements_total',
    'Total concept mastery achievements',
    ['concept', 'student_id']
)

# Pydantic models for API requests/responses
class TutoringSessionRequest(BaseModel):
    student_id: str = Field(..., description="Unique student identifier")
    target_concept: str = Field(..., description="Physics concept to learn")
    session_duration_minutes: int = Field(default=30, description="Expected session duration")
    student_preferences: Optional[Dict[str, Any]] = Field(default=None, description="Student learning preferences")

class TutoringSessionResponse(BaseModel):
    session_id: str
    target_concept: str
    initial_difficulty: float
    learning_path: List[str]
    prerequisites_met: bool
    missing_prerequisites: List[str]
    estimated_duration: int
    personalization_applied: bool

class StudentResponseRequest(BaseModel):
    session_id: str = Field(..., description="Active session identifier")
    problem_id: str = Field(..., description="Problem identifier")
    student_answer: str = Field(..., description="Student's answer")
    response_time_seconds: float = Field(..., description="Time taken to respond")
    is_correct: bool = Field(..., description="Whether answer is correct")
    engagement_data: Optional[Dict[str, Any]] = Field(default=None, description="Engagement metrics")
    
class AdaptiveProblemResponse(BaseModel):
    problem_id: str
    content: str
    difficulty: float
    problem_type: str
    hints: List[str]
    estimated_time_minutes: float
    learning_objectives: List[str]
    mastery_context: Dict[str, Any]
    math_scaffolding: Dict[str, Any]
    experiments_suggested: List[Dict[str, Any]]
    adaptation_metadata: Dict[str, Any]

class StudentProgressResponse(BaseModel):
    student_id: str
    overall_progress: Dict[str, Any]
    concept_masteries: Dict[str, float]
    learning_profile: Dict[str, Any]
    current_gaps: List[str]
    next_ready_concepts: List[str]
    recommendations: Dict[str, Any]
    privacy_protected: bool

class InterventionResponse(BaseModel):
    intervention_type: str
    content: str
    urgency: float
    timing: str
    personalization_data: Dict[str, Any]

# FastAPI app configuration
app = FastAPI(
    title="Physics Assistant Adaptive Tutoring API",
    description="Real-time intelligent tutoring system with personalized learning",
    version="6.2.0",
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

# Global tutoring engine instance
tutoring_engine: Optional[IntelligentTutoringEngine] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the intelligent tutoring engine on startup"""
    global tutoring_engine
    try:
        logger.info("🚀 Starting Phase 6.2 Adaptive Tutoring API Server")
        
        # Initialize tutoring engine
        tutoring_engine = IntelligentTutoringEngine()
        success = await tutoring_engine.initialize()
        
        if not success:
            raise Exception("Failed to initialize tutoring engine")
        
        logger.info("✅ Adaptive Tutoring API Server started successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to start tutoring API server: {e}")
        raise

@app.middleware("http")
async def performance_monitoring_middleware(request: Request, call_next):
    """Monitor API performance and ensure <200ms response times"""
    start_time = time.time()
    
    response = await call_next(request)
    
    # Record metrics
    duration = time.time() - start_time
    endpoint = request.url.path
    
    # Track adaptation response times
    if "/adaptive" in endpoint:
        ADAPTATION_TIME.labels(operation_type=endpoint.split('/')[-1]).observe(duration)
        
        # Log warning if response time exceeds target
        if duration > 0.2:  # 200ms target
            logger.warning(f"⚠️ Slow response on {endpoint}: {duration*1000:.1f}ms")
    
    return response

@app.post("/tutoring/session/start", response_model=TutoringSessionResponse)
async def start_tutoring_session(request: TutoringSessionRequest):
    """Start a new adaptive tutoring session"""
    try:
        start_time = time.time()
        
        if not tutoring_engine:
            raise HTTPException(status_code=500, detail="Tutoring engine not initialized")
        
        # Start adaptive session
        session_id = await tutoring_engine.start_adaptive_session(
            request.student_id,
            request.target_concept,
            request.session_duration_minutes
        )
        
        if not session_id:
            raise HTTPException(status_code=500, detail="Failed to create tutoring session")
        
        # Get session details
        session = tutoring_engine.active_sessions.get(session_id)
        student_state = tutoring_engine.student_knowledge_states.get(request.student_id)
        
        if not session or not student_state:
            raise HTTPException(status_code=500, detail="Session initialization failed")
        
        # Check prerequisites
        readiness = await tutoring_engine.mastery_tracker.check_mastery_readiness(
            request.student_id, request.target_concept, student_state
        )
        
        # Generate learning path
        learning_path = await tutoring_engine.concept_dependency_engine.get_learning_path(
            student_state.concept_masteries, request.target_concept
        )
        
        # Record metrics
        TUTORING_REQUESTS.labels(endpoint="start_session", student_id=request.student_id).inc()
        ACTIVE_SESSIONS.inc()
        
        response = TutoringSessionResponse(
            session_id=session_id,
            target_concept=session.target_concept,
            initial_difficulty=session.current_difficulty,
            learning_path=learning_path,
            prerequisites_met=readiness['ready'],
            missing_prerequisites=readiness.get('missing_prerequisites', []),
            estimated_duration=request.session_duration_minutes,
            personalization_applied=True
        )
        
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"🎯 Started tutoring session {session_id} in {processing_time:.1f}ms")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to start tutoring session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tutoring/session/{session_id}/problem", response_model=AdaptiveProblemResponse)
async def get_adaptive_problem(session_id: str):
    """Get next adaptive problem for the session"""
    try:
        start_time = time.time()
        
        if not tutoring_engine:
            raise HTTPException(status_code=500, detail="Tutoring engine not initialized")
        
        # Get enhanced adaptive problem
        enhanced_problem = await tutoring_engine.get_enhanced_adaptive_problem(session_id)
        
        if not enhanced_problem:
            raise HTTPException(status_code=404, detail="No problem available for session")
        
        problem_data = enhanced_problem['problem']
        
        response = AdaptiveProblemResponse(
            problem_id=problem_data['problem_id'],
            content=problem_data['content'],
            difficulty=problem_data['difficulty'],
            problem_type=problem_data['problem_type'],
            hints=problem_data['hints'],
            estimated_time_minutes=problem_data['estimated_time_minutes'],
            learning_objectives=problem_data['learning_objectives'],
            mastery_context=enhanced_problem['mastery_context'],
            math_scaffolding=enhanced_problem['math_scaffolding'],
            experiments_suggested=enhanced_problem['experiments_suggested'],
            adaptation_metadata=enhanced_problem['adaptation_metadata']
        )
        
        # Record metrics
        TUTORING_REQUESTS.labels(endpoint="get_problem", student_id="session").inc()
        
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"🧮 Generated adaptive problem in {processing_time:.1f}ms")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get adaptive problem: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tutoring/session/response")
async def process_student_response(request: StudentResponseRequest):
    """Process student response with real-time adaptation"""
    try:
        start_time = time.time()
        
        if not tutoring_engine:
            raise HTTPException(status_code=500, detail="Tutoring engine not initialized")
        
        # Process enhanced student response
        response_data = await tutoring_engine.process_enhanced_student_response(
            request.session_id,
            request.problem_id,
            request.student_answer,
            request.response_time_seconds,
            request.is_correct,
            request.engagement_data
        )
        
        if 'error' in response_data:
            raise HTTPException(status_code=404, detail=response_data['error'])
        
        # Record mastery achievements
        if response_data.get('mastery_achieved'):
            session = tutoring_engine.active_sessions.get(request.session_id)
            if session:
                MASTERY_ACHIEVEMENTS.labels(
                    concept=session.target_concept,
                    student_id=session.student_id
                ).inc()
        
        # Record metrics
        TUTORING_REQUESTS.labels(endpoint="process_response", student_id="session").inc()
        
        processing_time = (time.time() - start_time) * 1000
        response_data['processing_time_ms'] = processing_time
        
        logger.info(f"📊 Processed student response in {processing_time:.1f}ms")
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to process student response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tutoring/student/{student_id}/progress", response_model=StudentProgressResponse)
async def get_student_progress(student_id: str):
    """Get comprehensive student progress and recommendations"""
    try:
        start_time = time.time()
        
        if not tutoring_engine:
            raise HTTPException(status_code=500, detail="Tutoring engine not initialized")
        
        # Get enhanced progress summary
        progress_data = await tutoring_engine.get_learning_progress_summary(student_id)
        
        if not progress_data:
            raise HTTPException(status_code=404, detail="Student not found")
        
        # Get current gaps and ready concepts
        current_gaps = await tutoring_engine._get_current_gaps(student_id)
        next_ready = await tutoring_engine._get_next_ready_concepts(student_id)
        
        response = StudentProgressResponse(
            student_id=student_id,
            overall_progress=progress_data['overall_progress'],
            concept_masteries=progress_data['concept_masteries'],
            learning_profile=progress_data['learning_profile'],
            current_gaps=current_gaps,
            next_ready_concepts=next_ready,
            recommendations=progress_data['recommendations'],
            privacy_protected=True
        )
        
        # Record metrics
        TUTORING_REQUESTS.labels(endpoint="get_progress", student_id=student_id).inc()
        
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"📈 Generated progress report in {processing_time:.1f}ms")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get student progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tutoring/session/{session_id}/interventions")
async def get_real_time_interventions(session_id: str):
    """Get real-time intervention recommendations"""
    try:
        start_time = time.time()
        
        if not tutoring_engine:
            raise HTTPException(status_code=500, detail="Tutoring engine not initialized")
        
        session = tutoring_engine.active_sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        student_state = tutoring_engine.student_knowledge_states.get(session.student_id)
        if not student_state:
            raise HTTPException(status_code=404, detail="Student state not found")
        
        # Monitor for interventions
        problem_data = {
            'start_time': time.time() - 60,  # Assume 1 minute default
            'difficulty': session.current_difficulty,
            'engagement': session.engagement_score
        }
        
        interventions = await tutoring_engine.intervention_engine.monitor_and_trigger_interventions(
            session.student_id, student_state, session, problem_data
        )
        
        intervention_responses = [
            InterventionResponse(
                intervention_type=i.intervention_type.value,
                content=i.content,
                urgency=i.urgency,
                timing=i.timing,
                personalization_data=i.personalization_data
            )
            for i in interventions
        ]
        
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"🚨 Generated {len(interventions)} interventions in {processing_time:.1f}ms")
        
        return {
            'session_id': session_id,
            'interventions': intervention_responses,
            'processing_time_ms': processing_time
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get interventions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tutoring/session/{session_id}/end")
async def end_tutoring_session(session_id: str):
    """End a tutoring session and provide summary"""
    try:
        if not tutoring_engine:
            raise HTTPException(status_code=500, detail="Tutoring engine not initialized")
        
        session = tutoring_engine.active_sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Calculate session summary
        session_duration = (datetime.now() - session.start_time).total_seconds() / 60
        success_rate = session.problems_correct / max(1, session.problems_attempted)
        
        session_summary = {
            'session_id': session_id,
            'student_id': session.student_id,
            'target_concept': session.target_concept,
            'duration_minutes': session_duration,
            'problems_attempted': session.problems_attempted,
            'problems_correct': session.problems_correct,
            'success_rate': success_rate,
            'final_difficulty': session.current_difficulty,
            'engagement_score': session.engagement_score,
            'interventions_triggered': len(session.interventions_triggered),
            'adaptation_events': len(session.adaptation_events)
        }
        
        # Mark session as inactive
        session.is_active = False
        
        # Update metrics
        ACTIVE_SESSIONS.dec()
        
        logger.info(f"🏁 Ended tutoring session {session_id} - Duration: {session_duration:.1f}min, Success: {success_rate:.2f}")
        
        return session_summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to end tutoring session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tutoring/analytics/performance")
async def get_system_performance():
    """Get real-time system performance metrics"""
    try:
        if not tutoring_engine:
            raise HTTPException(status_code=500, detail="Tutoring engine not initialized")
        
        # Calculate performance metrics
        avg_response_time = 0
        if tutoring_engine.response_times:
            avg_response_time = sum(tutoring_engine.response_times) / len(tutoring_engine.response_times)
        
        performance_metrics = {
            'average_response_time_ms': avg_response_time,
            'target_response_time_ms': tutoring_engine.performance_target_ms,
            'meeting_target': avg_response_time <= tutoring_engine.performance_target_ms,
            'active_sessions': len([s for s in tutoring_engine.active_sessions.values() if s.is_active]),
            'total_students': len(tutoring_engine.student_knowledge_states),
            'cache_efficiency': len(tutoring_engine.inference_cache.get('learning_style_predictions', {})),
            'system_health': "healthy" if avg_response_time <= tutoring_engine.performance_target_ms else "degraded"
        }
        
        return performance_metrics
        
    except Exception as e:
        logger.error(f"❌ Failed to get system performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tutoring/concepts/dependencies")
async def get_concept_dependencies():
    """Get physics concept dependency graph"""
    try:
        if not tutoring_engine:
            raise HTTPException(status_code=500, detail="Tutoring engine not initialized")
        
        # Extract concept graph data
        concept_graph = tutoring_engine.concept_dependency_engine.concept_graph
        
        concepts = []
        for node in concept_graph.nodes(data=True):
            concept_name, attributes = node
            concepts.append({
                'name': concept_name,
                'difficulty': attributes.get('difficulty', 0.5),
                'category': attributes.get('category', 'general'),
                'physics_domain': attributes.get('physics_domain', 'general'),
                'prerequisites': list(concept_graph.predecessors(concept_name))
            })
        
        return {
            'concepts': concepts,
            'total_concepts': len(concepts),
            'dependency_edges': len(concept_graph.edges())
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get concept dependencies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'tutoring_engine_initialized': tutoring_engine is not None,
            'api_version': '6.2.0'
        }
        
        if tutoring_engine:
            health_status['active_sessions'] = len([s for s in tutoring_engine.active_sessions.values() if s.is_active])
            health_status['total_students'] = len(tutoring_engine.student_knowledge_states)
        
        return health_status
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return {'status': 'unhealthy', 'error': str(e)}

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "adaptive_tutoring_api:app",
        host="0.0.0.0",
        port=8002,
        reload=False,
        log_level="info"
    )