"""
FastAPI server for Physics Assistant API
All physics agents now use Strands SDK with MCP tools and Ollama LLM
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os

from hitl_knowledge_transfer import KnowledgeTransferGate

# Import all Strands-based agents
from strands_agents import (
    # Physics 101
    ForcesAgent,
    KinematicsAgent,
    MathAgent,
    MomentumAgent,
    EnergyAgent,
    AngularMotionAgent,
    # Physics 102
    ThermodynamicsAgent,
    WavesAgent,
    # Physics 201
    ElectromagnetismAgent,
    # Physics 202
    OpticsAgent,
    ModernPhysicsAgent,
    # Base class for type hints
    StrandsPhysicsAgent
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Valid agent IDs by course level - ALL use Strands SDK now
PHYSICS_101_AGENTS = ["forces_agent", "kinematics_agent", "math_agent", "momentum_agent", "energy_agent", "angular_motion_agent"]
PHYSICS_102_AGENTS = ["thermodynamics_agent", "waves_agent"]
PHYSICS_201_AGENTS = ["electromagnetism_agent"]
PHYSICS_202_AGENTS = ["optics_agent", "modern_physics_agent"]
ALL_VALID_AGENTS = PHYSICS_101_AGENTS + PHYSICS_102_AGENTS + PHYSICS_201_AGENTS + PHYSICS_202_AGENTS

# Global agent store for managing active agents (all Strands-based now)
agent_store: Dict[str, StrandsPhysicsAgent] = {}
knowledge_transfer_gate: Optional[KnowledgeTransferGate] = None

# Pydantic models for API requests/responses
class AgentCreateRequest(BaseModel):
    """Request model for creating a physics agent"""
    agent_id: str = Field(
        ...,
        description="Agent type: Physics 101 (forces, kinematics, math, momentum, energy, angular_motion) or Physics 102-202 (thermodynamics, waves, electromagnetism, optics, modern_physics)",
        pattern="^(forces_agent|kinematics_agent|math_agent|momentum_agent|energy_agent|angular_motion_agent|thermodynamics_agent|waves_agent|electromagnetism_agent|optics_agent|modern_physics_agent)$"
    )
    use_direct_tools: bool = Field(
        default=True, 
        description="Whether to use direct MCP tools (recommended)"
    )
    enable_rag: bool = Field(
        default=True,
        description="Whether to enable RAG context augmentation"
    )
    rag_api_url: Optional[str] = Field(
        default=None,
        description="URL for RAG API server (defaults to DATABASE_API_HOST env var)"
    )

class AgentCreateResponse(BaseModel):
    """Response model for agent creation"""
    success: bool
    agent_id: str
    message: str
    capabilities: Optional[Dict[str, Any]] = None

class ProblemSolveRequest(BaseModel):
    """Request model for solving physics problems"""
    problem: str = Field(..., description="Physics problem description")
    context: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Optional context for the problem"
    )
    user_id: Optional[str] = Field(
        default="api_user",
        description="User identifier for database logging"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session identifier for database logging"
    )

class ProblemSolveResponse(BaseModel):
    """Response model for problem solving"""
    success: bool
    agent_id: str
    problem: str
    solution: Optional[str] = None
    reasoning: Optional[str] = None
    tools_used: Optional[list] = None
    execution_time_ms: Optional[int] = None
    diagram: Optional[Dict[str, Any]] = None
    hitl: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class AgentHealthResponse(BaseModel):
    """Response model for agent health check"""
    agent_id: str
    status: str
    tools_count: int
    ready: bool
    mode: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global knowledge_transfer_gate
    logger.info("🚀 Starting Physics Assistant API")
    database_api_host = os.getenv("DATABASE_API_HOST", "localhost")
    database_api_port = os.getenv("DATABASE_API_PORT", "8001")
    database_api_url = f"http://{database_api_host}:{database_api_port}"
    knowledge_transfer_gate = KnowledgeTransferGate(
        database_api_url=database_api_url,
    )
    yield
    logger.info("🛑 Shutting down Physics Assistant API")
    if knowledge_transfer_gate:
        await knowledge_transfer_gate.cleanup()
        knowledge_transfer_gate = None
    # Cleanup agents if needed
    agent_store.clear()

# Create FastAPI application
app = FastAPI(
    title="Physics Assistant API",
    description="API for physics tutoring agents with MCP tool integration",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper functions
def _get_strands_agent_class(agent_id: str):
    """Get the Strands agent class for a given agent_id"""
    strands_agent_map = {
        # Physics 101
        "forces_agent": ForcesAgent,
        "kinematics_agent": KinematicsAgent,
        "math_agent": MathAgent,
        "momentum_agent": MomentumAgent,
        "energy_agent": EnergyAgent,
        "angular_motion_agent": AngularMotionAgent,
        # Physics 102
        "thermodynamics_agent": ThermodynamicsAgent,
        "waves_agent": WavesAgent,
        # Physics 201
        "electromagnetism_agent": ElectromagnetismAgent,
        # Physics 202
        "optics_agent": OpticsAgent,
        "modern_physics_agent": ModernPhysicsAgent,
    }
    return strands_agent_map.get(agent_id)


async def get_or_create_agent(agent_id: str, use_direct_tools: bool = True, enable_rag: bool = True, rag_api_url: str = None):
    """Get existing agent or create new one with RAG and database logging enabled

    All agents now use Strands SDK with MCP tools and Ollama LLM.
    The use_direct_tools parameter is kept for API compatibility but ignored.
    """
    import os
    database_api_host = os.getenv("DATABASE_API_HOST", "localhost")
    database_api_port = os.getenv("DATABASE_API_PORT", "8001")
    database_api_url = f"http://{database_api_host}:{database_api_port}"

    if rag_api_url is None:
        rag_api_url = database_api_url

    # Simplified key - all agents use Strands now
    agent_key = f"{agent_id}_{enable_rag}"

    if agent_key not in agent_store:
        logger.info(f"Creating new Strands agent: {agent_id} (RAG: {'enabled' if enable_rag else 'disabled'})")

        strands_class = _get_strands_agent_class(agent_id)

        if not strands_class:
            raise ValueError(f"Unknown agent_id: {agent_id}")

        agent = strands_class(
            database_api_url=database_api_url,
            enable_database_logging=True,
            enable_rag=enable_rag,
            rag_api_url=rag_api_url
        )

        await agent.initialize()
        agent_store[agent_key] = agent
        logger.info(f"Agent {agent_id} created (Strands SDK) with RAG: {'enabled' if enable_rag else 'disabled'}")

    return agent_store[agent_key]


def _append_perf_stage(
    stages: list[Dict[str, Any]],
    stage: str,
    started_at: float,
    **extra: Any,
) -> None:
    duration_ms = int((time.perf_counter() - started_at) * 1000)
    item: Dict[str, Any] = {"stage": stage, "duration_ms": duration_ms}
    item.update(extra)
    stages.append(item)


def _finalize_perf_trace(component: str, stages: list[Dict[str, Any]], started_at: float) -> Dict[str, Any]:
    total_ms = int((time.perf_counter() - started_at) * 1000)
    slowest = max(stages, key=lambda x: int(x.get("duration_ms", 0)), default={"stage": "none", "duration_ms": 0})
    return {
        "component": component,
        "total_ms": total_ms,
        "slowest_stage": slowest.get("stage", "none"),
        "slowest_duration_ms": int(slowest.get("duration_ms", 0)),
        "stages": stages,
    }


def _merge_performance_trace(
    api_trace: Dict[str, Any],
    hitl_trace: Optional[Dict[str, Any]] = None,
    agent_trace: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {"api": api_trace}
    if isinstance(hitl_trace, dict) and hitl_trace:
        merged["hitl"] = hitl_trace
    if isinstance(agent_trace, dict) and agent_trace:
        merged["agent"] = agent_trace

    candidates: list[Dict[str, Any]] = []
    for component_key in ("api", "hitl", "agent"):
        trace = merged.get(component_key)
        if not isinstance(trace, dict):
            continue
        for stage in trace.get("stages", []) or []:
            if not isinstance(stage, dict):
                continue
            candidates.append(
                {
                    "component": component_key,
                    "stage": str(stage.get("stage", "unknown")),
                    "duration_ms": int(stage.get("duration_ms", 0)),
                }
            )

    slowest = max(candidates, key=lambda x: int(x.get("duration_ms", 0)), default={"component": "none", "stage": "none", "duration_ms": 0})
    merged["request_total_ms"] = int(api_trace.get("total_ms", 0))
    merged["slowest_stage"] = slowest
    return merged


def _log_slow_trace(agent_id: str, user_id: Optional[str], performance_trace: Dict[str, Any]) -> None:
    total_ms = int(performance_trace.get("request_total_ms", 0))
    threshold_ms = int(os.getenv("PERF_SLOW_REQUEST_MS", "15000"))
    if total_ms < threshold_ms:
        return
    slowest = performance_trace.get("slowest_stage") or {}
    logger.warning(
        "SLOW_REQUEST_TRACE agent=%s user=%s total_ms=%s slowest_component=%s slowest_stage=%s slowest_ms=%s",
        agent_id,
        user_id or "unknown_user",
        total_ms,
        slowest.get("component", "unknown"),
        slowest.get("stage", "unknown"),
        int(slowest.get("duration_ms", 0)),
    )

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Physics Assistant API",
        "version": "2.0.0",
        "docs_url": "/docs",
        "available_agents": ALL_VALID_AGENTS,
        "framework": "strands",
        "courses": {
            "physics_101": PHYSICS_101_AGENTS,
            "physics_102": PHYSICS_102_AGENTS,
            "physics_201": PHYSICS_201_AGENTS,
            "physics_202": PHYSICS_202_AGENTS
        }
    }

@app.post("/agent/create", response_model=AgentCreateResponse)
async def create_agent(request: AgentCreateRequest) -> AgentCreateResponse:
    """
    Create and initialize a physics agent using Strands SDK

    All agents now use Strands SDK with MCP tools and Ollama LLM.
    """
    try:
        logger.info(f"Creating agent: {request.agent_id}")
        
        # Get or create agent using the existing function
        agent = await get_or_create_agent(request.agent_id, request.use_direct_tools, request.enable_rag, request.rag_api_url)
        
        # Get capabilities
        capabilities = await agent.get_capabilities()
        
        return AgentCreateResponse(
            success=True,
            agent_id=request.agent_id,
            message=f"Agent {request.agent_id} created and initialized successfully",
            capabilities=capabilities
        )
        
    except Exception as e:
        logger.error(f"Error creating agent {request.agent_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create agent: {str(e)}"
        )

@app.post("/agent/{agent_id}/solve", response_model=ProblemSolveResponse)
async def solve_problem(
    agent_id: str,
    request: ProblemSolveRequest,
    use_direct_tools: bool = True,
    enable_rag: bool = True,
    rag_api_url: Optional[str] = None
) -> ProblemSolveResponse:
    """
    Solve a physics problem using the specified agent
    """
    request_started = time.perf_counter()
    api_trace_stages: list[Dict[str, Any]] = []
    hitl_trace: Optional[Dict[str, Any]] = None

    try:
        stage_started = time.perf_counter()
        # Validate agent_id
        if agent_id not in ALL_VALID_AGENTS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid agent_id: {agent_id}. Must be one of: {', '.join(ALL_VALID_AGENTS)}"
            )
        _append_perf_stage(api_trace_stages, "validate_agent_id", stage_started)
        
        effective_problem = request.problem
        hitl_question = None
        hitl_result = None
        guidance_prefix = None

        kt_response = None
        class_identifier = None
        if isinstance(request.context, dict):
            kt_response = request.context.get("knowledge_transfer_response")
            class_identifier = request.context.get("class_identifier")

        if knowledge_transfer_gate and knowledge_transfer_gate.is_enabled_for(agent_id):
            if isinstance(kt_response, dict):
                check_id = str(kt_response.get("check_id", "")).strip()
                selected_option_id = str(kt_response.get("selected_option_id", "")).strip()
                if check_id and selected_option_id:
                    stage_started = time.perf_counter()
                    hitl_result, hitl_trace = knowledge_transfer_gate.process_answer_with_trace(
                        check_id=check_id,
                        selected_option_id=selected_option_id,
                        user_id=request.user_id or "react_user",
                        session_id=request.session_id,
                        class_identifier=class_identifier,
                    )
                    _append_perf_stage(
                        api_trace_stages,
                        "hitl_process_answer",
                        stage_started,
                        answer_processed=bool(hitl_result),
                    )
                    if hitl_result:
                        effective_problem = str(hitl_result.get("original_problem", request.problem))
                        guidance_prefix = str(hitl_result.get("guidance", "")).strip()
                    else:
                        logger.error(
                            "HITL_GATE_FALLBACK: invalid/expired check_id provided; proceeding without gating. "
                            "agent=%s user=%s check_id=%s",
                            agent_id,
                            request.user_id,
                            check_id,
                        )
                else:
                    logger.error(
                        "HITL_GATE_FALLBACK: malformed knowledge_transfer_response payload; proceeding without gating. "
                        "agent=%s user=%s",
                        agent_id,
                        request.user_id,
                    )
            else:
                stage_started = time.perf_counter()
                hitl_question, hitl_trace = await knowledge_transfer_gate.maybe_create_check_with_trace(
                    agent_id=agent_id,
                    problem=request.problem,
                    user_id=request.user_id or "react_user",
                    session_id=request.session_id,
                    class_identifier=class_identifier,
                )
                _append_perf_stage(
                    api_trace_stages,
                    "hitl_maybe_create_check",
                    stage_started,
                    question_required=bool(hitl_question),
                )
                if hitl_question:
                    stage_started = time.perf_counter()
                    _append_perf_stage(api_trace_stages, "early_return_hitl_question", stage_started)
                    api_trace = _finalize_perf_trace("api_route", api_trace_stages, request_started)
                    performance_trace = _merge_performance_trace(api_trace=api_trace, hitl_trace=hitl_trace)
                    _log_slow_trace(agent_id, request.user_id, performance_trace)
                    return ProblemSolveResponse(
                        success=True,
                        agent_id=agent_id,
                        problem=request.problem,
                        solution="",
                        hitl=hitl_question,
                        metadata={
                            "hitl": hitl_question,
                            "framework": "strands",
                            "performance_trace": performance_trace,
                        },
                    )

        # Get or create agent
        stage_started = time.perf_counter()
        agent = await get_or_create_agent(agent_id, use_direct_tools, enable_rag, rag_api_url)
        _append_perf_stage(api_trace_stages, "get_or_create_agent", stage_started)

        logger.info(f"Solving problem with {agent_id}: {effective_problem[:50]}...")

        # Solve the problem with user and session context for database logging
        stage_started = time.perf_counter()
        result = await agent.solve_problem(
            problem=effective_problem,
            context=request.context,
            user_id=request.user_id,
            session_id=request.session_id
        )
        _append_perf_stage(
            api_trace_stages,
            "agent_solve_problem",
            stage_started,
            success=bool(result.get("success")),
        )

        if result.get("success") and guidance_prefix and result.get("solution"):
            result["solution"] = f"{guidance_prefix}\n\n{result['solution']}"

        stage_started = time.perf_counter()
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        agent_trace = metadata.get("performance_trace") if isinstance(metadata.get("performance_trace"), dict) else None
        if hitl_result:
            metadata["hitl"] = {
                "status": "answer_processed",
                "check_id": hitl_result.get("check_id"),
                "concept_tag": hitl_result.get("concept_tag"),
                "was_correct": hitl_result.get("was_correct"),
                "confidence": hitl_result.get("confidence"),
                "threshold": hitl_result.get("threshold"),
            }
            result["hitl"] = metadata["hitl"]
        elif hitl_question:
            metadata["hitl"] = hitl_question
            result["hitl"] = hitl_question
        _append_perf_stage(api_trace_stages, "response_metadata_assembly", stage_started)

        api_trace = _finalize_perf_trace("api_route", api_trace_stages, request_started)
        metadata["performance_trace"] = _merge_performance_trace(
            api_trace=api_trace,
            hitl_trace=hitl_trace,
            agent_trace=agent_trace,
        )
        _log_slow_trace(agent_id, request.user_id, metadata["performance_trace"])
        if metadata:
            result["metadata"] = metadata

        return ProblemSolveResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error solving problem with {agent_id}: {str(e)}")
        api_trace = _finalize_perf_trace("api_route", api_trace_stages, request_started)
        performance_trace = _merge_performance_trace(api_trace=api_trace, hitl_trace=hitl_trace)
        _log_slow_trace(agent_id, request.user_id, performance_trace)
        return ProblemSolveResponse(
            success=False,
            agent_id=agent_id,
            problem=request.problem,
            error=str(e),
            metadata={"performance_trace": performance_trace},
        )

@app.get("/agent/{agent_id}/health", response_model=AgentHealthResponse)
async def check_agent_health(
    agent_id: str,
    use_direct_tools: bool = True
) -> AgentHealthResponse:
    """
    Check the health status of a physics agent
    """
    try:
        # Validate agent_id
        if agent_id not in ALL_VALID_AGENTS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid agent_id: {agent_id}. Must be one of: {', '.join(ALL_VALID_AGENTS)}"
            )
        
        # Get or create agent (use defaults for RAG)
        agent = await get_or_create_agent(agent_id, use_direct_tools)

        # Get health status
        health = await agent.health_check()
        
        return AgentHealthResponse(**health)
        
    except Exception as e:
        logger.error(f"Error checking health for {agent_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check agent health: {str(e)}"
        )

@app.get("/agent/{agent_id}/capabilities")
async def get_agent_capabilities(
    agent_id: str,
    use_direct_tools: bool = True
) -> Dict[str, Any]:
    """
    Get the capabilities of a physics agent
    """
    try:
        # Validate agent_id
        if agent_id not in ALL_VALID_AGENTS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid agent_id: {agent_id}. Must be one of: {', '.join(ALL_VALID_AGENTS)}"
            )
        
        # Get or create agent (use defaults for RAG)
        agent = await get_or_create_agent(agent_id, use_direct_tools)

        # Get capabilities
        capabilities = await agent.get_capabilities()
        
        return capabilities
        
    except Exception as e:
        logger.error(f"Error getting capabilities for {agent_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get agent capabilities: {str(e)}"
        )

@app.get("/agents/list")
async def list_available_agents():
    """
    List all available physics agents organized by course level
    """
    return {
        "available_agents": [
            # Physics 101 - Mechanics (LangChain-based)
            {
                "agent_id": "forces_agent",
                "name": "Forces Agent",
                "description": "Handles force analysis, free body diagrams, and Newton's laws",
                "course": "Physics 101",
                "framework": "langchain"
            },
            {
                "agent_id": "kinematics_agent",
                "name": "Kinematics Agent",
                "description": "Handles motion analysis, projectile motion, and kinematics equations",
                "course": "Physics 101",
                "framework": "langchain"
            },
            {
                "agent_id": "math_agent",
                "name": "Math Agent",
                "description": "Handles mathematical calculations, algebra, and computational problems",
                "course": "All",
                "framework": "langchain"
            },
            {
                "agent_id": "momentum_agent",
                "name": "Momentum Agent",
                "description": "Handles momentum, impulse, and collision problems",
                "course": "Physics 101",
                "framework": "langchain"
            },
            {
                "agent_id": "energy_agent",
                "name": "Energy Agent",
                "description": "Handles work, energy, power, and conservation of energy problems",
                "course": "Physics 101",
                "framework": "langchain"
            },
            {
                "agent_id": "angular_motion_agent",
                "name": "Angular Motion Agent",
                "description": "Handles rotational motion, angular momentum, and torque problems",
                "course": "Physics 101",
                "framework": "langchain"
            },
            # Physics 102 - Thermodynamics & Waves (Strands-based)
            {
                "agent_id": "thermodynamics_agent",
                "name": "Thermodynamics Agent",
                "description": "Handles ideal gas law, heat transfer, thermal expansion, and Carnot efficiency",
                "course": "Physics 102",
                "framework": "strands"
            },
            {
                "agent_id": "waves_agent",
                "name": "Waves Agent",
                "description": "Handles wave mechanics, Doppler effect, sound intensity, and interference",
                "course": "Physics 102",
                "framework": "strands"
            },
            # Physics 201 - Electricity & Magnetism (Strands-based)
            {
                "agent_id": "electromagnetism_agent",
                "name": "Electromagnetism Agent",
                "description": "Handles Coulomb's law, circuits, magnetic fields, and Faraday's law",
                "course": "Physics 201",
                "framework": "strands"
            },
            # Physics 202 - Optics & Modern Physics (Strands-based)
            {
                "agent_id": "optics_agent",
                "name": "Optics Agent",
                "description": "Handles refraction, lenses, mirrors, diffraction, and interference",
                "course": "Physics 202",
                "framework": "strands"
            },
            {
                "agent_id": "modern_physics_agent",
                "name": "Modern Physics Agent",
                "description": "Handles relativity, quantum mechanics, and nuclear physics",
                "course": "Physics 202",
                "framework": "strands"
            }
        ],
        "active_agents": list(agent_store.keys()),
        "agent_counts": {
            "physics_101": len(PHYSICS_101_AGENTS),
            "physics_102": len(PHYSICS_102_AGENTS),
            "physics_201": len(PHYSICS_201_AGENTS),
            "physics_202": len(PHYSICS_202_AGENTS),
            "total": len(ALL_VALID_AGENTS)
        }
    }
# @app.get("/agents/list")
# async def list_available_agents():
#     return {
#         "available_agents": list(Config.PHYSICS_AGENTS.values()),
#         "active_agents": list(agent_store.keys())
#     }

@app.delete("/agent/{agent_id}")
async def remove_agent(agent_id: str, use_direct_tools: bool = True):
    """
    Remove an agent from the store
    """
    agent_key = f"{agent_id}_{use_direct_tools}"
    
    if agent_key in agent_store:
        del agent_store[agent_key]
        return {"message": f"Agent {agent_id} removed successfully"}
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Agent {agent_id} not found"
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    API health check endpoint
    """
    return {
        "status": "healthy",
        "active_agents": len(agent_store),
        "agent_keys": list(agent_store.keys())
    }

# RAG Configuration Endpoints
@app.get("/rag/status")
async def get_rag_status():
    """
    Get RAG system status and configuration
    """
    try:
        # Check how many agents have RAG enabled
        rag_enabled_agents = [
            key for key in agent_store.keys() 
            if key.split("_")[-1] == "True"  # enable_rag is the last part of the key
        ]
        
        return {
            "status": "available",
            "rag_enabled_agents": len(rag_enabled_agents),
            "total_agents": len(agent_store),
            "rag_endpoints": [
                "/rag/query",
                "/rag/semantic-search", 
                "/rag/graph-search",
                "/rag/learning-path"
            ]
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/rag/clear-cache")
async def clear_rag_cache():
    """
    Clear RAG client caches for all agents
    """
    try:
        cleared_count = 0
        for agent in agent_store.values():
            if hasattr(agent, 'rag_client') and agent.rag_client:
                agent.rag_client.clear_cache()
                cleared_count += 1
        
        return {
            "status": "success",
            "message": f"RAG cache cleared for {cleared_count} agents"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear RAG cache: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
