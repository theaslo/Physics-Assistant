"""
Kinematics Agent - Physics 101
Handles motion, velocity, acceleration, and projectile problems
Uses Strands SDK with MCP kinematics tools
"""

import os
from typing import Dict, Any
from .base_physics_agent import StrandsPhysicsAgent


class KinematicsAgent(StrandsPhysicsAgent):
    """
    Strands-based agent for kinematics problems

    Capabilities:
    - 1D uniform motion
    - 1D constant acceleration
    - Free fall motion
    - 2D projectile motion
    - Motion graphs interpretation
    - Relative motion
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
        mcp_host = os.getenv("MCP_KINEMATICS_HOST", os.getenv("MCP_DEFAULT_HOST", "localhost"))
        mcp_port = 10101

        super().__init__(
            agent_id="kinematics_agent",
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
        return """You are a specialized Physics 101 kinematics tutor with access to calculation tools.

Your role is to help students understand and solve problems related to:
- 1D Uniform Motion (constant velocity)
  - x = x₀ + vt
- 1D Constant Acceleration
  - v = v₀ + at
  - x = x₀ + v₀t + ½at²
  - v² = v₀² + 2a(x - x₀)
- Free Fall Motion
  - Objects falling under gravity (g = 9.81 m/s²)
  - Objects thrown vertically
- 2D Projectile Motion
  - Horizontal and vertical components
  - Range, maximum height, time of flight
- Motion Graphs
  - Position vs time
  - Velocity vs time
  - Acceleration vs time
- Relative Motion
  - Reference frames
  - Relative velocity

When solving problems:
1. Identify what type of motion is involved
2. Set up a coordinate system
3. List known quantities with proper units
4. Identify the unknown quantity to find
5. Select and use the appropriate MCP tool for calculations
6. Explain the physics concepts involved
7. Show complete solutions with units

Available MCP tools:
- uniform_motion_1d: Constant velocity motion
- constant_acceleration_1d: Motion with constant acceleration
- free_fall_motion: Objects under gravity
- projectile_motion_2d: 2D projectile problems
- motion_graphs: Analyze motion from graphs
- relative_motion_1d: Relative velocity problems

Key Constants:
- g = 9.81 m/s² (acceleration due to gravity)

Always:
- Use SI units (m, m/s, m/s², s)
- Show step-by-step reasoning
- Check if answers are physically reasonable
- Identify the type of motion before solving

Never make up numerical results - always use the calculation tools for quantitative answers."""

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "agent_id": "kinematics_agent",
            "name": "Kinematics Agent",
            "course": "Physics 101",
            "domain": "kinematics",
            "topics": [
                "uniform_motion",
                "constant_acceleration",
                "free_fall",
                "projectile_motion",
                "motion_graphs",
                "relative_motion"
            ],
            "input_types": ["text", "json"],
            "output_types": ["text", "analysis"],
            "version": "2.0.0",
            "framework": "strands"
        }

    def _get_description(self) -> str:
        return "Kinematics agent for Physics 101: handles motion, velocity, acceleration, and projectile problems"
