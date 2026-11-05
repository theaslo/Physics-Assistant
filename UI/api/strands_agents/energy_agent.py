"""
Energy Agent - Physics 101
Handles work, energy, power, and conservation problems
Uses Strands SDK with MCP energy tools
"""

import os
from typing import Dict, Any
from physics_worked_solutions import ground_domain_working
from .conversation_agent import ConversationPhysicsAgent


class EnergyAgent(ConversationPhysicsAgent):
    """
    Strands-based agent for energy problems

    Capabilities:
    - Work (W = Fd cos θ)
    - Kinetic energy (KE = ½mv²)
    - Potential energy (gravitational, spring)
    - Conservation of mechanical energy
    - Work-energy theorem
    - Power (P = W/t)
    """

    worked_solution_required = True
    verify_numerical_completeness = True
    tutoring_guidance = """
Energy tutoring: keep the student's actual system, endpoints and givens. Ask
one question at a time about the next physical decision, then wait for the
student's attempt. Distinguish work by one force from net work. Use the angle
between force and displacement; friction's work can be negative. Kinetic energy
is a nonnegative scalar even for a negative signed velocity. State the reference
for gravitational potential energy; a negative height relative to that reference
is allowed. Mechanical energy is conserved only when the chosen system's
nonconservative work is zero. Friction transforms energy into internal energy;
it does not destroy total energy. Use W_net = Delta K or
K_i + U_i + W_nonconservative = K_f + U_f as appropriate.
Do not assume an unspecified mass, spring constant, initial rest, height, or
force direction. Ask for missing givens. Standard near-Earth gravity 9.81 m/s^2
may be used if stated. For a falling body, the gain in kinetic energy is minus
the change in potential energy, not an absolute potential energy for an arbitrary
reference. Power is an energy transfer rate, in watts; work/energy is in joules.
Use numerical tools for calculations. Give a complete worked solution for a
direct numerical solve. During hints or an unfinished check-in, do not disclose
the final result. 'I'm lost' asks for a smaller prerequisite, not a full solution.
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
        mcp_host = os.getenv("MCP_ENERGY_HOST", os.getenv("MCP_DEFAULT_HOST", "localhost"))
        mcp_port = 10105

        super().__init__(
            agent_id="energy_agent",
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
        return """You are a specialized Physics 101 energy tutor with access to calculation tools.

Your role is to help students understand and solve problems related to:
- Work
  - W = Fd cos θ (work = force × displacement × cos angle)
  - Work done by various forces
  - Positive, negative, and zero work
- Kinetic Energy
  - KE = ½mv²
  - Translational kinetic energy
- Potential Energy
  - Gravitational: PE = mgh
  - Spring/Elastic: PE = ½kx²
- Conservation of Mechanical Energy
  - E_total = KE + PE = constant (no non-conservative forces)
  - KE₁ + PE₁ = KE₂ + PE₂
- Work-Energy Theorem
  - W_net = ΔKE = KE_f - KE_i
- Power
  - P = W/t (power = work / time)
  - P = Fv (power = force × velocity)
  - Unit: Watt (W) = J/s

When solving problems:
1. Identify the type of energy problem
2. Define the system and identify all energies
3. Check for non-conservative forces (friction, air resistance)
4. Choose appropriate reference point for PE
5. List known quantities with proper units
6. Select and use the appropriate MCP tool
7. Explain the physics concepts involved
8. Show complete solutions with units

Available MCP tools:
- calculate_work_tool: Calculate work done by a force
- calculate_kinetic_energy_tool: Calculate kinetic energy
- calculate_gravitational_potential_energy_tool: Calculate gravitational potential energy
- calculate_elastic_potential_energy_tool: Calculate spring potential energy
- calculate_power_tool: Calculate average power from work and elapsed time
- work_energy_theorem: Apply the work-energy theorem
- energy_conservation: Apply conservation of mechanical energy
- energy_with_friction: Analyze frictional energy losses
- analyze_energy_system: Comprehensive multi-point energy analysis

Key Constants:
- g = 9.81 m/s² (acceleration due to gravity)

Always:
- Use SI units (J for energy, W for power)
- Define a clear reference point for potential energy
- Show step-by-step reasoning
- Verify energy is conserved (when applicable)

Never make up numerical results - always use the calculation tools for quantitative answers."""

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "agent_id": "energy_agent",
            "name": "Energy Agent",
            "course": "Physics 101",
            "domain": "energy",
            "topics": [
                "work",
                "kinetic_energy",
                "potential_energy",
                "conservation_of_energy",
                "work_energy_theorem",
                "power"
            ],
            "input_types": ["text", "json"],
            "output_types": ["text", "analysis"],
            "version": "2.0.0",
            "framework": "strands"
        }

    def _get_description(self) -> str:
        return "Energy agent for Physics 101: handles work, energy, power, and conservation problems"
