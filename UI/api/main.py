"""
FastAPI server for Physics Assistant API
All physics agents now use Strands SDK with MCP tools and Ollama LLM
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from physics_graphs import (
    build_kinematics_graph_response,
    build_kinematics_graphs_from_context,
    should_attempt_kinematics_graph,
)

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
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class KinematicsGraphRequest(BaseModel):
    """Request model for generating 1D kinematics graphs without invoking an agent"""
    problem: Optional[str] = Field(
        default="",
        description="Natural-language kinematics problem or graph request"
    )
    inputs: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured kinematics values such as x0, v0, a, t, x, v, or time_range"
    )

class KinematicsGraphResponse(BaseModel):
    """Response model for reusable graph payloads"""
    success: bool
    graphs: list
    warnings: list
    errors: list
    parsed_inputs: Dict[str, Any]

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
    logger.info("🚀 Starting Physics Assistant API")
    yield
    logger.info("🛑 Shutting down Physics Assistant API")
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


def _format_graph_value(value: Any, unit: str = "") -> str:
    """Format graph-derived values for student-facing fallback solutions."""
    if isinstance(value, (int, float)):
        if abs(value) >= 100:
            formatted = f"{value:.1f}"
        elif abs(value) >= 10:
            formatted = f"{value:.2f}"
        else:
            formatted = f"{value:.3f}".rstrip("0").rstrip(".")
    else:
        formatted = str(value)

    return f"{formatted} {unit}".strip()


def _parameter_map(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        parameter.get("symbol"): parameter
        for parameter in payload.get("parameters", [])
        if parameter.get("symbol")
    }


def _parameter_value(parameters: Dict[str, Dict[str, Any]], symbol: str) -> str:
    parameter = parameters[symbol]
    return _format_graph_value(parameter.get("value"), parameter.get("unit", ""))


def _build_kinematics_fallback_solution(graph_result: Dict[str, Any]) -> Optional[str]:
    """Create a deterministic solution when the LLM is unavailable.

    The graph builder has already parsed the prompt and solved the same values
    used by the rendered graph, so this keeps the text answer and graph aligned.
    """
    graph_payloads = graph_result.get("graphs") or []
    if not graph_payloads:
        return None

    payload = graph_payloads[0]
    parameters = _parameter_map(payload)
    motion_type = payload.get("motionType")

    if motion_type in {"uniform_motion", "constant_acceleration"} and all(
        symbol in parameters for symbol in ("x0", "v0", "a", "t", "v", "dx")
    ):
        x0 = _parameter_value(parameters, "x0")
        v0 = _parameter_value(parameters, "v0")
        acceleration = _parameter_value(parameters, "a")
        duration = _parameter_value(parameters, "t")
        final_velocity = _parameter_value(parameters, "v")
        displacement = _parameter_value(parameters, "dx")

        return (
            "**Solution**\n\n"
            f"Known values: x0 = {x0}, v0 = {v0}, a = {acceleration}, t = {duration}.\n\n"
            "Use the constant-acceleration equations:\n\n"
            "- v = v0 + at\n"
            "- dx = v0*t + 0.5*a*t^2\n\n"
            f"Final velocity: v = {v0} + ({acceleration})({duration}) = {final_velocity}.\n\n"
            f"Displacement: dx = ({v0})({duration}) + 0.5({acceleration})({duration})^2 = {displacement}.\n\n"
            f"Final answer: after {duration}, the car's velocity is {final_velocity} and it has traveled {displacement}."
        )

    if motion_type == "projectile_motion" and all(
        symbol in parameters for symbol in ("v0", "theta", "h0", "g", "T", "R", "hmax")
    ):
        launch_speed = _parameter_value(parameters, "v0")
        angle = _parameter_value(parameters, "theta")
        height = _parameter_value(parameters, "h0")
        gravity = _parameter_value(parameters, "g")
        flight_time = _parameter_value(parameters, "T")
        range_value = _parameter_value(parameters, "R")
        max_height = _parameter_value(parameters, "hmax")

        return (
            "**Solution**\n\n"
            f"Known values: launch speed = {launch_speed}, angle = {angle}, initial height = {height}, g = {gravity}.\n\n"
            "Use projectile motion:\n\n"
            "- x = x0 + v0x*t\n"
            "- y = h0 + v0y*t - 0.5*g*t^2\n"
            "- vy = v0y - g*t\n\n"
            f"Final answer: flight time is {flight_time}, horizontal range is {range_value}, "
            f"and maximum height is {max_height}."
        )

    if motion_type == "piecewise_kinematics" and all(
        symbol in parameters for symbol in ("stages", "T", "x final", "v final")
    ):
        stages = _parameter_value(parameters, "stages")
        total_time = _parameter_value(parameters, "T")
        final_position = _parameter_value(parameters, "x final")
        final_velocity = _parameter_value(parameters, "v final")

        return (
            "**Solution**\n\n"
            f"The motion has {stages}. The final state is computed by applying each stage in order, "
            "using the previous stage's final position and velocity as the next stage's initial values.\n\n"
            f"Final answer: total time is {total_time}, final position is {final_position}, "
            f"and final velocity is {final_velocity}."
        )

    return None

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
    try:
        # Validate agent_id
        if agent_id not in ALL_VALID_AGENTS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid agent_id: {agent_id}. Must be one of: {', '.join(ALL_VALID_AGENTS)}"
            )

        if agent_id == "kinematics_agent":
            graph_result = build_kinematics_graphs_from_context(request.problem, request.context)
            fallback_solution = _build_kinematics_fallback_solution(graph_result)
            if fallback_solution:
                return ProblemSolveResponse(
                    success=True,
                    agent_id=agent_id,
                    problem=request.problem,
                    solution=fallback_solution,
                    reasoning="Used deterministic kinematics equations to keep the solution and graph aligned.",
                    tools_used=["kinematics_graph_builder"],
                    metadata={
                        "graphs": graph_result["graphs"],
                        "graph_warnings": graph_result["warnings"],
                        "graph_errors": graph_result["errors"],
                        "graph_inputs": graph_result["parsed_inputs"],
                    },
                )
        
        # Get or create agent
        agent = await get_or_create_agent(agent_id, use_direct_tools, enable_rag, rag_api_url)
        
        logger.info(f"Solving problem with {agent_id}: {request.problem[:50]}...")
        
        # Solve the problem with user and session context for database logging
        result = await agent.solve_problem(
            problem=request.problem, 
            context=request.context,
            user_id=request.user_id,
            session_id=request.session_id
        )

        if agent_id == "kinematics_agent":
            graph_result = build_kinematics_graphs_from_context(request.problem, request.context)
            metadata = dict(result.get("metadata") or {})
            metadata["graphs"] = graph_result["graphs"]
            metadata["graph_warnings"] = graph_result["warnings"]
            metadata["graph_errors"] = graph_result["errors"]
            metadata["graph_inputs"] = graph_result["parsed_inputs"]
            result["metadata"] = metadata

            if not result.get("solution") and graph_result["graphs"]:
                fallback_solution = _build_kinematics_fallback_solution(graph_result)
                if fallback_solution:
                    if result.get("error"):
                        metadata["agent_error"] = result["error"]
                    result.update(
                        {
                            "success": True,
                            "solution": fallback_solution,
                            "reasoning": "Used deterministic kinematics equations to keep the solution and graph aligned.",
                            "tools_used": result.get("tools_used") or ["kinematics_graph_builder"],
                            "error": None,
                        }
                    )

            if (
                not graph_result["graphs"]
                and graph_result["errors"]
                and should_attempt_kinematics_graph(request.problem, request.context)
                and result.get("solution")
            ):
                result["solution"] += "\n\nGraph note: " + " ".join(graph_result["errors"])
        
        return ProblemSolveResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error solving problem with {agent_id}: {str(e)}")
        return ProblemSolveResponse(
            success=False,
            agent_id=agent_id,
            problem=request.problem,
            error=str(e)
        )

@app.post("/kinematics/graphs", response_model=KinematicsGraphResponse)
async def create_kinematics_graphs(request: KinematicsGraphRequest) -> KinematicsGraphResponse:
    """
    Generate reusable 1D kinematics graph payloads from a student question or structured inputs.
    """
    graph_result = build_kinematics_graph_response(
        problem=request.problem or "",
        structured=request.inputs,
    )

    return KinematicsGraphResponse(
        success=len(graph_result["errors"]) == 0 and len(graph_result["graphs"]) > 0,
        graphs=graph_result["graphs"],
        warnings=graph_result["warnings"],
        errors=graph_result["errors"],
        parsed_inputs=graph_result["parsed_inputs"],
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
