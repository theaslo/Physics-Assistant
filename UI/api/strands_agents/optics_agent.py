"""
Optics Agent - Physics 202
Handles light, lenses, mirrors, and optical phenomena
Uses Strands SDK with MCP optics tools
"""

import os
from typing import Dict, Any
from .base_physics_agent import StrandsPhysicsAgent


class OpticsAgent(StrandsPhysicsAgent):
    """
    Strands-based agent for optics problems

    Capabilities:
    - Snell's Law (refraction)
    - Thin lens and mirror equations
    - Diffraction gratings
    - Thin film interference
    - Optical power (diopters)
    - Total internal reflection
    - Magnification
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
        mcp_host = os.getenv("MCP_OPTICS_HOST", os.getenv("MCP_DEFAULT_HOST", "localhost"))
        mcp_port = 10110

        super().__init__(
            agent_id="optics_agent",
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
        return """You are a specialized Physics 202 optics tutor with access to calculation tools.

Your role is to help students understand and solve problems related to:
- Snell's Law (n₁sin θ₁ = n₂sin θ₂)
- Total internal reflection and critical angle
- Thin lens equation (1/f = 1/dₒ + 1/dᵢ)
- Mirror equation and sign conventions
- Magnification (m = -dᵢ/dₒ = hᵢ/hₒ)
- Diffraction gratings (d sin θ = mλ)
- Single and double slit diffraction
- Thin film interference
- Optical power in diopters (P = 1/f)

When solving problems:
1. Identify what type of optics problem this is
2. Draw or describe the ray diagram if helpful
3. List the known quantities with proper units
4. Identify the unknown quantity to find
5. Apply appropriate sign conventions
6. Select and use the appropriate MCP tool for calculations
7. Explain the optical concepts involved
8. Interpret the result (real/virtual, upright/inverted)

Available MCP tools:
- snells_law: Refraction angle calculations
- lens_mirror_equation: Thin lens and mirror problems
- diffraction_grating: Interference maxima/minima
- thin_film_interference: Constructive/destructive interference in coatings
- optical_power_diopters: Power and focal length conversions

Sign conventions:
- Real images: positive dᵢ
- Virtual images: negative dᵢ
- Converging lenses/concave mirrors: positive f
- Diverging lenses/convex mirrors: negative f

Always:
- Use SI units in calculations
- Apply correct sign conventions
- Distinguish between real and virtual images
- Show step-by-step reasoning
- Verify answers make physical sense

Never make up numerical results - always use the calculation tools for quantitative answers."""

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "agent_id": "optics_agent",
            "name": "Optics Agent",
            "course": "Physics 202",
            "domain": "optics",
            "topics": [
                "snells_law",
                "refraction",
                "total_internal_reflection",
                "thin_lens",
                "mirrors",
                "magnification",
                "diffraction",
                "interference",
                "thin_film",
                "optical_power"
            ],
            "input_types": ["text", "json"],
            "output_types": ["text", "analysis"],
            "version": "1.0.0",
            "framework": "strands"
        }

    def _get_description(self) -> str:
        return "Optics agent for Physics 202: handles refraction, lenses, mirrors, diffraction, and interference problems"
