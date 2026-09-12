"""
Forces Agent - Physics 101
Handles Newton's laws, springs, friction, and equilibrium problems
Uses Strands SDK with MCP forces tools
"""

import os
from typing import Dict, Any
from .base_physics_agent import StrandsPhysicsAgent


class ForcesAgent(StrandsPhysicsAgent):
    """
    Strands-based agent for forces problems

    Capabilities:
    - Newton's Second Law (F=ma)
    - Spring force (Hooke's Law)
    - Friction (static and kinetic)
    - Force components and vectors
    - Equilibrium problems
    - Inclined plane problems
    - Tension and pulley systems
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
        mcp_host = os.getenv("MCP_FORCES_HOST", os.getenv("MCP_DEFAULT_HOST", "localhost"))
        mcp_port = 10100

        super().__init__(
            agent_id="forces_agent",
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
        return """You are a specialized Physics 101 forces tutor with access to calculation tools.

Your role is to help students understand and solve problems related to:
- Newton's Laws of Motion
  - First Law (Inertia)
  - Second Law (F = ma)
  - Third Law (Action-Reaction)
- Spring Forces (Hooke's Law: F = -kx)
- Friction Forces
  - Static friction (f_s ≤ μ_s N)
  - Kinetic friction (f_k = μ_k N)
- Force Components and Vectors
- Equilibrium (ΣF = 0)
- Inclined Plane Problems
- Tension and Pulley Systems

When solving problems:
1. Identify all forces acting on the object(s)
2. Draw or describe a free body diagram
3. Choose an appropriate coordinate system
4. List known quantities with proper units
5. Identify the unknown quantity to find
6. Select and use the appropriate MCP tool for calculations
7. Explain the physics concepts involved
8. Show complete solutions with units

Available MCP tools:
- newton_second_law: Calculate force, mass, or acceleration using F=ma
- calculate_spring_force_tool: Calculate spring force using Hooke's Law
- calculate_friction_force_tool: Calculate static or kinetic friction
- resolve_force_components: Resolve forces into components
- check_equilibrium: Analyze forces in equilibrium
- analyze_forces_on_incline: Solve inclined plane problems
- analyze_tension_forces: Solve rope/string tension and pulley systems
- create_free_body_diagram: Generate free-body force breakdowns
- add_forces_2d: Add multiple 2D force vectors

Always:
- Use SI units (N, kg, m/s²)
- Show step-by-step reasoning
- Draw attention to common misconceptions
- Verify answers make physical sense

Never make up numerical results - always use the calculation tools for quantitative answers."""

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "agent_id": "forces_agent",
            "name": "Forces Agent",
            "course": "Physics 101",
            "domain": "forces",
            "topics": [
                "newtons_laws",
                "spring_force",
                "friction",
                "force_components",
                "equilibrium",
                "inclined_plane",
                "tension",
                "pulleys"
            ],
            "input_types": ["text", "json"],
            "output_types": ["text", "analysis"],
            "version": "2.0.0",
            "framework": "strands"
        }

    def _get_description(self) -> str:
        return "Forces agent for Physics 101: handles Newton's laws, springs, friction, and equilibrium problems"
