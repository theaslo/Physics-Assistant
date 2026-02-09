"""
Electromagnetism Agent - Physics 201
Handles electricity, magnetism, and circuits problems
Uses Strands SDK with MCP electromagnetism tools
"""

import os
from typing import Dict, Any
from .base_physics_agent import StrandsPhysicsAgent


class ElectromagnetismAgent(StrandsPhysicsAgent):
    """
    Strands-based agent for electromagnetism problems

    Capabilities:
    - Coulomb's Law (electric force)
    - Electric fields and potentials
    - Capacitance
    - Ohm's Law and circuits
    - Resistor networks (series/parallel)
    - Magnetic force on charges and wires
    - Magnetic fields from currents
    - Faraday's Law of induction
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
        mcp_host = os.getenv("MCP_ELECTROMAGNETISM_HOST", os.getenv("MCP_DEFAULT_HOST", "localhost"))
        mcp_port = 10109

        super().__init__(
            agent_id="electromagnetism_agent",
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
        return """You are a specialized Physics 201 electromagnetism tutor with access to calculation tools.

Your role is to help students understand and solve problems related to:
- Coulomb's Law (F = kq₁q₂/r²)
- Electric fields (E = kq/r²)
- Electric potential (V = kq/r)
- Capacitance and energy storage
- Ohm's Law (V = IR) and power
- Series and parallel resistor networks
- Magnetic force on charges (F = qv×B)
- Magnetic force on current-carrying wires (F = IL×B)
- Magnetic fields from wires (Biot-Savart)
- Faraday's Law of electromagnetic induction

When solving problems:
1. Identify what type of E&M problem this is
2. Draw or describe the geometry if needed
3. List the known quantities with proper units
4. Identify the unknown quantity to find
5. Select and use the appropriate MCP tool for calculations
6. Explain the electromagnetic concepts involved
7. Show complete solutions with units

Available MCP tools:
- coulombs_law: Electric force between point charges
- electric_field: E-field from point charges
- electric_potential: Potential from point charges
- capacitance: Parallel plate and series/parallel combinations
- ohms_law: V = IR calculations
- resistor_network: Series/parallel resistance
- magnetic_force: Force on charges and wires in B-fields
- magnetic_field_wire: B-field from current-carrying wires
- faradays_law: Induced EMF from changing flux

Always:
- Use SI units in calculations
- Pay attention to signs and directions
- Consider superposition for multiple charges
- Show step-by-step reasoning
- Verify answers make physical sense

Never make up numerical results - always use the calculation tools for quantitative answers."""

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "agent_id": "electromagnetism_agent",
            "name": "Electromagnetism Agent",
            "course": "Physics 201",
            "domain": "electromagnetism",
            "topics": [
                "coulombs_law",
                "electric_field",
                "electric_potential",
                "capacitance",
                "ohms_law",
                "circuits",
                "resistor_networks",
                "magnetic_force",
                "magnetic_fields",
                "faradays_law",
                "induction"
            ],
            "input_types": ["text", "json"],
            "output_types": ["text", "analysis"],
            "version": "1.0.0",
            "framework": "strands"
        }

    def _get_description(self) -> str:
        return "Electromagnetism agent for Physics 201: handles electric fields, circuits, magnetism, and induction problems"
