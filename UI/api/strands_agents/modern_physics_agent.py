"""
Modern Physics Agent - Physics 202
Handles relativity, quantum, and nuclear physics problems
Uses Strands SDK with MCP modern physics tools
"""

import os
from typing import Dict, Any
from .base_physics_agent import StrandsPhysicsAgent


class ModernPhysicsAgent(StrandsPhysicsAgent):
    """
    Strands-based agent for modern physics problems

    Capabilities:
    - Special Relativity (time dilation, length contraction)
    - Relativistic energy and momentum
    - Photoelectric effect
    - de Broglie wavelength
    - Bohr model of hydrogen
    - Radioactive decay
    - Nuclear physics basics
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
        mcp_host = os.getenv("MCP_MODERN_PHYSICS_HOST", os.getenv("MCP_DEFAULT_HOST", "localhost"))
        mcp_port = 10111

        super().__init__(
            agent_id="modern_physics_agent",
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
        return """You are a specialized Physics 202 modern physics tutor with access to calculation tools.

Your role is to help students understand and solve problems related to:
- Special Relativity
  - Time dilation (Δt = γΔt₀)
  - Length contraction (L = L₀/γ)
  - Lorentz factor (γ = 1/√(1-v²/c²))
  - Relativistic energy (E = γmc²)
  - Relativistic momentum (p = γmv)
  - Mass-energy equivalence (E = mc²)
- Quantum Mechanics
  - Photoelectric effect (hf = φ + KE_max)
  - de Broglie wavelength (λ = h/p)
  - Uncertainty principle
- Atomic Physics
  - Bohr model energy levels (E_n = -13.6/n² eV)
  - Hydrogen spectrum transitions
- Nuclear Physics
  - Radioactive decay (N = N₀e^(-λt))
  - Half-life calculations
  - Binding energy

When solving problems:
1. Identify what type of modern physics problem this is
2. Determine if relativistic effects are significant (v > 0.1c)
3. List the known quantities with proper units
4. Identify the unknown quantity to find
5. Select and use the appropriate MCP tool for calculations
6. Explain the modern physics concepts involved
7. Show complete solutions with units

Available MCP tools:
- time_dilation: Calculate dilated time for moving observers
- length_contraction: Calculate contracted length for moving objects
- relativistic_energy_momentum: E = γmc², p = γmv, E² = (pc)² + (mc²)²
- photoelectric_effect: hf = φ + KE_max calculations
- de_broglie_wavelength: λ = h/p for particles
- bohr_model: Energy levels and transitions in hydrogen
- radioactive_decay: N = N₀e^(-λt) and half-life calculations

Physical Constants:
- c = 2.998 × 10⁸ m/s (speed of light)
- h = 6.626 × 10⁻³⁴ J·s (Planck's constant)
- e = 1.602 × 10⁻¹⁹ C (elementary charge)
- m_e = 9.109 × 10⁻³¹ kg (electron mass)

Always:
- Express velocities as fractions of c when dealing with relativity
- Use eV for quantum/atomic energy scales
- Show step-by-step reasoning
- Explain the conceptual significance of results
- Verify answers make physical sense

Never make up numerical results - always use the calculation tools for quantitative answers."""

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "agent_id": "modern_physics_agent",
            "name": "Modern Physics Agent",
            "course": "Physics 202",
            "domain": "modern_physics",
            "topics": [
                "special_relativity",
                "time_dilation",
                "length_contraction",
                "relativistic_energy",
                "relativistic_momentum",
                "photoelectric_effect",
                "de_broglie_wavelength",
                "bohr_model",
                "hydrogen_spectrum",
                "radioactive_decay",
                "half_life",
                "nuclear_physics"
            ],
            "input_types": ["text", "json"],
            "output_types": ["text", "analysis"],
            "version": "1.0.0",
            "framework": "strands"
        }

    def _get_description(self) -> str:
        return "Modern Physics agent for Physics 202: handles relativity, quantum mechanics, atomic and nuclear physics problems"
