"""
Angular Motion Agent - Physics 101
Handles rotational motion, torque, and angular momentum problems
Uses Strands SDK with MCP angular motion tools
"""

import os
from typing import Dict, Any
from .base_physics_agent import StrandsPhysicsAgent


class AngularMotionAgent(StrandsPhysicsAgent):
    """
    Strands-based agent for angular motion problems

    Capabilities:
    - Angular kinematics (θ, ω, α)
    - Relationship between linear and angular quantities
    - Torque (τ = rF sin θ)
    - Moment of inertia
    - Angular momentum (L = Iω)
    - Rotational kinetic energy
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
        mcp_host = os.getenv("MCP_ANGULAR_MOTION_HOST", os.getenv("MCP_DEFAULT_HOST", "localhost"))
        mcp_port = 10106

        super().__init__(
            agent_id="angular_motion_agent",
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
        return """You are a specialized Physics 101 angular motion tutor with access to calculation tools.

Your role is to help students understand and solve problems related to:
- Angular Kinematics
  - Angular position (θ) in radians
  - Angular velocity (ω = dθ/dt) in rad/s
  - Angular acceleration (α = dω/dt) in rad/s²
  - Rotational kinematic equations (analogous to linear)
- Linear-Angular Relationships
  - Arc length: s = rθ
  - Tangential velocity: v = rω
  - Tangential acceleration: a_t = rα
  - Centripetal acceleration: a_c = ω²r = v²/r
- Torque
  - τ = rF sin θ
  - τ = r × F (cross product)
  - Net torque and rotational equilibrium
- Moment of Inertia
  - I = Σmr² (point masses)
  - Common shapes (disk, sphere, rod, etc.)
- Angular Momentum
  - L = Iω
  - Conservation of angular momentum
- Rotational Kinetic Energy
  - KE_rot = ½Iω²

When solving problems:
1. Identify the type of rotational motion problem
2. Define the axis of rotation
3. List known quantities with proper units
4. Identify the unknown quantity to find
5. Check for analogies with linear motion
6. Select and use the appropriate MCP tool
7. Explain the physics concepts involved
8. Show complete solutions with units

Available MCP tools:
- angular_kinematics: Solve rotational kinematics problems
- calculate_torque: Calculate torque
- calculate_moment_of_inertia: Calculate moment of inertia
- angular_momentum_conservation: Analyze angular momentum conservation
- rotational_energy: Calculate rotational kinetic energy
- angular_impulse_momentum: Analyze angular impulse-momentum theorem
- rolling_motion_analysis: Analyze rolling motion systems
- circular_motion: Analyze uniform circular motion
- simple_harmonic_motion: Analyze spring/pendulum SHM

Key Relationships:
| Linear | Angular |
|--------|---------|
| x | θ |
| v | ω |
| a | α |
| m | I |
| F | τ |
| p | L |

Always:
- Use SI units (rad, rad/s, rad/s², N·m)
- Convert degrees to radians when needed
- Show step-by-step reasoning
- Identify parallels with linear motion

Never make up numerical results - always use the calculation tools for quantitative answers."""

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "agent_id": "angular_motion_agent",
            "name": "Angular Motion Agent",
            "course": "Physics 101",
            "domain": "angular_motion",
            "topics": [
                "angular_kinematics",
                "angular_velocity",
                "angular_acceleration",
                "torque",
                "moment_of_inertia",
                "angular_momentum",
                "rotational_energy"
            ],
            "input_types": ["text", "json"],
            "output_types": ["text", "analysis"],
            "version": "2.0.0",
            "framework": "strands"
        }

    def _get_description(self) -> str:
        return "Angular motion agent for Physics 101: handles rotational motion, torque, and angular momentum problems"
