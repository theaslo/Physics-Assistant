"""
Momentum Agent - Physics 101
Handles momentum, impulse, and collision problems
Uses Strands SDK with MCP momentum tools
"""

import os
from typing import Dict, Any
from .base_physics_agent import StrandsPhysicsAgent


class MomentumAgent(StrandsPhysicsAgent):
    """
    Strands-based agent for momentum problems

    Capabilities:
    - Linear momentum (p = mv)
    - Impulse (J = FΔt = Δp)
    - Conservation of momentum
    - Elastic collisions
    - Inelastic collisions
    - Perfectly inelastic collisions
    """

    def __init__(
        self,
        llm_host: str = "http://ds.stat.uconn.edu:11434",
        model_id: str = "qwen3:8b-q8_0",
        database_api_url: str = "http://localhost:8001",
        enable_database_logging: bool = True,
        enable_rag: bool = True,
        rag_api_url: str = "http://localhost:8001"
    ):
        # Get MCP host from environment for Docker compatibility
        mcp_host = os.getenv("MCP_MOMENTUM_HOST", os.getenv("MCP_DEFAULT_HOST", "localhost"))
        mcp_port = 10104

        super().__init__(
            agent_id="momentum_agent",
            mcp_host=mcp_host,
            mcp_port=mcp_port,
            llm_host=llm_host,
            model_id=model_id,
            database_api_url=database_api_url,
            enable_database_logging=enable_database_logging,
            enable_rag=enable_rag,
            rag_api_url=rag_api_url
        )

    def _get_system_prompt(self) -> str:
        return """You are a specialized Physics 101 momentum tutor with access to calculation tools.

Your role is to help students understand and solve problems related to:
- Linear Momentum
  - p = mv (momentum = mass × velocity)
  - Momentum is a vector quantity
- Impulse
  - J = FΔt (impulse = force × time)
  - J = Δp (impulse = change in momentum)
  - Impulse-Momentum Theorem
- Conservation of Momentum
  - In isolated systems: Σp_before = Σp_after
  - m₁v₁ᵢ + m₂v₂ᵢ = m₁v₁f + m₂v₂f
- Elastic Collisions
  - Both momentum and kinetic energy conserved
  - Objects bounce off each other
- Inelastic Collisions
  - Only momentum conserved
  - Some kinetic energy lost
- Perfectly Inelastic Collisions
  - Objects stick together after collision
  - Maximum kinetic energy loss

When solving problems:
1. Identify the type of momentum problem
2. Define the system and check if it's isolated
3. List known quantities (masses, velocities)
4. Identify what type of collision (if applicable)
5. Select and use the appropriate MCP tool
6. Explain the physics concepts involved
7. Show complete solutions with units

Available MCP tools:
- calculate_momentum_1d: Calculate 1D momentum p = mv
- calculate_momentum_2d: Calculate 2D momentum vectors
- calculate_impulse_1d: Calculate 1D impulse
- calculate_impulse_2d: Calculate 2D impulse vectors
- momentum_impulse_theorem: Apply J = Δp
- momentum_conservation_1d: Solve 1D collisions
- momentum_conservation_2d: Solve 2D collisions
- analyze_collision: Comprehensive collision analysis

Key Concepts:
- Momentum is conserved in all collisions (isolated system)
- Kinetic energy only conserved in elastic collisions
- Direction matters (momentum is a vector)

Always:
- Use SI units (kg·m/s for momentum)
- Pay attention to velocity signs (direction)
- Show step-by-step reasoning
- Verify momentum is conserved

Never make up numerical results - always use the calculation tools for quantitative answers."""

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "agent_id": "momentum_agent",
            "name": "Momentum Agent",
            "course": "Physics 101",
            "domain": "momentum",
            "topics": [
                "linear_momentum",
                "impulse",
                "conservation_of_momentum",
                "elastic_collision",
                "inelastic_collision",
                "impulse_momentum_theorem"
            ],
            "input_types": ["text", "json"],
            "output_types": ["text", "analysis"],
            "version": "2.0.0",
            "framework": "strands"
        }

    def _get_description(self) -> str:
        return "Momentum agent for Physics 101: handles momentum, impulse, and collision problems"
