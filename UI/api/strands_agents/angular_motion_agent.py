"""
Angular Motion Agent - Physics 101
Handles rotational motion, torque, and angular momentum problems
Uses Strands SDK with MCP angular motion tools
"""

import os
from typing import Dict, Any
from physics_worked_solutions import ground_domain_working
from .conversation_agent import ConversationPhysicsAgent


class AngularMotionAgent(ConversationPhysicsAgent):
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

    worked_solution_required = True
    verify_numerical_completeness = True
    tutoring_guidance = """
Angular-motion tutoring: identify the rotation axis, the actual body/system,
and a consistent signed direction. Ask one next-step question and wait for the
student's attempt. Constant-angular-acceleration equations require constant
alpha; do not apply them to a time-dependent alpha without integration. Use
radians in linear/angular relations and convert degrees or rpm explicitly.
Torque is r cross F; a force through the axis gives zero torque. The angle is
between r and F, not automatically 90 degrees. Moment of inertia depends on
mass distribution and axis. tau_net = I alpha assumes constant I about a fixed
axis. Angular momentum is conserved only for negligible net external torque
about that axis. A skater can increase rotational kinetic energy by internal
work while conserving angular momentum; zero external torque does not imply
constant angular speed when I changes. Rolling without slipping requires
v_cm = R omega and includes translational and rotational kinetic energies.
Never invent an inertia, radius, mass, initial rest, or torque direction. Ask
for missing conditions. Use numerical tools for calculations and show complete
working for direct numerical solves. In guided reflection or an unfinished
check-in, give one hint without revealing the final result. 'I'm lost' asks for
a smaller prerequisite; release a full solution only on an explicit request.
"""

    def _ground_worked_solution(self, worked, tool_outputs):
        return ground_domain_working(worked, tool_outputs)

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
