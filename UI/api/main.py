"""
FastAPI server for Physics Assistant API
Hosts CombinedPhysicsAgent with dynamic agent selection
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from agent import CombinedPhysicsAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global agent store for managing active agents
agent_store: Dict[str, CombinedPhysicsAgent] = {}

# Pydantic models for API requests/responses
class AgentCreateRequest(BaseModel):
    """Request model for creating a physics agent"""
    agent_id: str = Field(
        ..., 
        description="Agent type: 'forces_agent' or 'kinematics_agent'",
        pattern="^(forces_agent|kinematics_agent)$"
    )
    use_direct_tools: bool = Field(
        default=True, 
        description="Whether to use direct MCP tools (recommended)"
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

class ProblemSolveResponse(BaseModel):
    """Response model for problem solving"""
    success: bool
    agent_id: str
    problem: str
    solution: Optional[str] = None
    reasoning: Optional[str] = None
    tools_used: Optional[list] = None
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
async def get_or_create_agent(agent_id: str, use_direct_tools: bool = True) -> CombinedPhysicsAgent:
    """Get existing agent or create new one"""
    agent_key = f"{agent_id}_{use_direct_tools}"
    
    if agent_key not in agent_store:
        logger.info(f"Creating new agent: {agent_id}")
        agent = CombinedPhysicsAgent(
            agent_id=agent_id,
            use_direct_tools=use_direct_tools
        )
        await agent.initialize()
        agent_store[agent_key] = agent
        logger.info(f"Agent {agent_id} created and initialized")
    
    return agent_store[agent_key]

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Physics Assistant API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "available_agents": ["forces_agent", "kinematics_agent"]
    }

@app.post("/agent/create", response_model=AgentCreateResponse)
async def create_agent(request: AgentCreateRequest) -> AgentCreateResponse:
    """
    Create and initialize a physics agent
    
    Compatible with the specified pattern:
    ```python
    return CombinedPhysicsAgent(
        agent_id=<USER REQUESTED AGENT>, 
        use_direct_tools=use_direct_tools
    )
    ```
    """
    try:
        logger.info(f"Creating agent: {request.agent_id}")
        
        # Create agent using the required pattern
        agent = CombinedPhysicsAgent(
            agent_id=request.agent_id,
            use_direct_tools=request.use_direct_tools
        )
        
        # Initialize the agent
        await agent.initialize()
        
        # Store in agent store
        agent_key = f"{request.agent_id}_{request.use_direct_tools}"
        agent_store[agent_key] = agent
        
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
    use_direct_tools: bool = True
) -> ProblemSolveResponse:
    """
    Solve a physics problem using the specified agent
    """
    try:
        # Validate agent_id
        if agent_id not in ["forces_agent", "kinematics_agent"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid agent_id: {agent_id}. Must be 'forces_agent' or 'kinematics_agent'"
            )
        
        # Get or create agent
        agent = await get_or_create_agent(agent_id, use_direct_tools)
        
        logger.info(f"Solving problem with {agent_id}: {request.problem[:50]}...")
        
        # Solve the problem
        result = await agent.solve_problem(request.problem, request.context)
        
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
        if agent_id not in ["forces_agent", "kinematics_agent"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid agent_id: {agent_id}"
            )
        
        # Get or create agent
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
        if agent_id not in ["forces_agent", "kinematics_agent"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid agent_id: {agent_id}"
            )
        
        # Get or create agent
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
    List all available physics agents
    """
    return {
        "available_agents": [
            {
                "agent_id": "forces_agent",
                "name": "Forces Agent",
                "description": "Handles force analysis, free body diagrams, and Newton's laws"
            },
            {
                "agent_id": "kinematics_agent", 
                "name": "Kinematics Agent",
                "description": "Handles motion analysis, projectile motion, and kinematics equations"
            }
        ],
        "active_agents": list(agent_store.keys())
    }

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )