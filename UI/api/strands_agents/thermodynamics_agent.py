"""
Thermodynamics Agent - Physics 102
Handles heat, temperature, ideal gas, and thermal physics problems
Uses Strands SDK with MCP thermodynamics tools
"""

import os
from typing import Dict, Any
from physics_worked_solutions import ground_domain_working
from .conversation_agent import ConversationPhysicsAgent


class ThermodynamicsAgent(ConversationPhysicsAgent):
    """
    Strands-based agent for thermodynamics problems

    Capabilities:
    - Ideal gas law calculations (PV = nRT)
    - Heat transfer (Q = mcΔT)
    - Thermal expansion
    - Heat conduction (Fourier's Law)
    - Carnot efficiency
    - First and Second Laws of Thermodynamics
    """

    worked_solution_required = True
    verify_numerical_completeness = True
    check_in_guidance = """
For thermodynamics, ask ONE decision about the energy transfer, process
constraint, or temperature convention that is needed for THIS requested
calculation. State the sign convention explicitly if testing a work sign.
Refer to the actual gas/process in the question. Prefer asking whether a
specified transfer adds or removes energy in THIS event, or what the stated
process holds constant. All choices must answer that SAME transfer or quantity.
Do not mix work BY and work ON the system in different choices. Do not ask
an unqualified definition of "the sign of work by a system": it depends on
expansion versus compression. State Delta U = Q - W_by when testing the sign.
Ask about the physical decision, not "which law applies" or an abstract definition.
Keep the answer choices at the same level: do not pit a specific first-law
equation against the general statement of energy conservation, since both
can be true. Do not imply an isothermal or adiabatic process unless stated.
Do not include MCP, tool names or the final numerical result in the check.
"""
    tutoring_guidance = """
Thermodynamics tutoring: identify the system, process, givens and sign convention.
Ask one next-step question and wait for the student's attempt. Use Delta U = Q - W,
with heat into the system positive and work BY the system positive. Work ON the
system is the negative of work by it. For an ideal gas, isothermal means constant
temperature and Delta U = 0, not Q = 0. Adiabatic means Q = 0, not constant
temperature. An isochoric process has zero boundary work, but can transfer heat.
Use P Delta V only at constant pressure. Heat and work are transfers, not stored
state properties. Q = mc Delta T assumes no phase change and the stated specific
heat; a temperature difference has the same numerical value in kelvin and Celsius.
Absolute gas-law and Carnot temperatures MUST be in kelvin. For Celsius absolute
temperatures convert T_K = T_C + 273.15 before those calculations. Carnot
efficiency 1 - T_c/T_h is an upper bound for an engine, not a promise for a real
engine; require T_h > T_c > 0. Do not assume a process, amount of gas, material's
specific heat, heat input, or reservoir temperature. Ask for missing information.
Use first_law_thermodynamics for Q, work or Delta U; use ideal_gas_law for states,
passing only known state quantities in gas_data. OMIT the unknown state variable
or use null: pressure=0 is not a placeholder for unknown pressure, and an ideal
gas at positive volume, amount and temperature cannot have zero absolute pressure.
Use heat_transfer for sensible heat, and constant_pressure_work only when constant
pressure is specified. Use numerical tools for every numerical result. Show
complete working for direct solves. In guided reflection or an unfinished
check-in, give one hint without disclosing the final answer. 'I'm lost' asks for
a smaller prerequisite, not a full solution; honor explicit full-solution requests.
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
        mcp_host = os.getenv("MCP_THERMODYNAMICS_HOST", os.getenv("MCP_DEFAULT_HOST", "localhost"))
        mcp_port = 10107

        super().__init__(
            agent_id="thermodynamics_agent",
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
        return """You are a specialized Physics 102 thermodynamics tutor with access to calculation tools.

Your role is to help students understand and solve problems related to:
- Ideal gas law (PV = nRT)
- Heat transfer and specific heat (Q = mcΔT)
- Thermal expansion (linear and volumetric)
- Heat conduction (Fourier's Law)
- Carnot efficiency and heat engines
- First Law of Thermodynamics (ΔU = Q - W)
- Second Law and entropy

When solving problems:
1. Identify what type of thermodynamics problem this is
2. List the known quantities with proper units
3. Identify the unknown quantity to find
4. Select and use the appropriate MCP tool for calculations
5. Explain the physics concepts involved
6. Show complete solutions with units

Available MCP tools:
- ideal_gas_law: PV = nRT calculations (find P, V, n, or T)
- heat_transfer: Q = mcΔT calculations
- thermal_expansion: Linear and volumetric expansion
- heat_conduction: Fourier's Law for heat flow
- carnot_efficiency: Maximum efficiency of heat engines
- first_law_thermodynamics: Calculate heat, work, or change in internal energy
- constant_pressure_work: Boundary work at explicitly constant pressure

Always:
- Use SI units in calculations
- Explain physical concepts clearly
- Show step-by-step reasoning
- Verify answers make physical sense

Never make up numerical results - always use the calculation tools for quantitative answers."""

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "agent_id": "thermodynamics_agent",
            "name": "Thermodynamics Agent",
            "course": "Physics 102",
            "domain": "thermodynamics",
            "topics": [
                "ideal_gas_law",
                "heat_transfer",
                "thermal_expansion",
                "heat_conduction",
                "carnot_efficiency",
                "first_law",
                "second_law",
                "entropy"
            ],
            "input_types": ["text", "json"],
            "output_types": ["text", "analysis"],
            "version": "1.0.0",
            "framework": "strands"
        }

    def _get_description(self) -> str:
        return "Thermodynamics agent for Physics 102: handles ideal gas, heat transfer, thermal expansion, and heat engine problems"
