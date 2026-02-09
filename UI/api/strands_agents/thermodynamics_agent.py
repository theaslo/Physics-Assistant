"""
Thermodynamics Agent - Physics 102
Handles heat, temperature, ideal gas, and thermal physics problems
Uses Strands SDK with MCP thermodynamics tools
"""

import os
from typing import Dict, Any
from .base_physics_agent import StrandsPhysicsAgent


class ThermodynamicsAgent(StrandsPhysicsAgent):
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
