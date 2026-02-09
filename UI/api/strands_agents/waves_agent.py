"""
Waves Agent - Physics 102
Handles wave mechanics, sound, and oscillation problems
Uses Strands SDK with MCP waves tools
"""

import os
from typing import Dict, Any
from .base_physics_agent import StrandsPhysicsAgent


class WavesAgent(StrandsPhysicsAgent):
    """
    Strands-based agent for wave physics problems

    Capabilities:
    - Wave equation (v = fλ)
    - Doppler effect
    - Sound intensity and decibels
    - Standing waves (strings and pipes)
    - Wave interference
    - Superposition principle
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
        mcp_host = os.getenv("MCP_WAVES_HOST", os.getenv("MCP_DEFAULT_HOST", "localhost"))
        mcp_port = 10108

        super().__init__(
            agent_id="waves_agent",
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
        return """You are a specialized Physics 102 waves and sound tutor with access to calculation tools.

Your role is to help students understand and solve problems related to:
- Wave properties (wavelength, frequency, velocity, amplitude)
- Wave equation v = fλ
- Doppler effect for sound and light
- Sound intensity and decibel scale
- Standing waves in strings and air columns
- Wave interference (constructive and destructive)
- Superposition principle

When solving problems:
1. Identify what type of wave problem this is
2. List the known quantities with proper units
3. Identify the unknown quantity to find
4. Select and use the appropriate MCP tool for calculations
5. Explain the wave physics concepts involved
6. Show complete solutions with units

Available MCP tools:
- wave_equation: v = fλ calculations
- doppler_effect: Frequency shifts for moving sources/observers
- sound_intensity_decibels: Intensity and decibel conversions
- standing_waves: Harmonics in strings and pipes
- wave_interference: Constructive/destructive interference analysis

Always:
- Use SI units in calculations
- Clarify whether dealing with transverse or longitudinal waves
- Distinguish between source and observer motion for Doppler
- Show step-by-step reasoning
- Verify answers make physical sense

Never make up numerical results - always use the calculation tools for quantitative answers."""

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "agent_id": "waves_agent",
            "name": "Waves Agent",
            "course": "Physics 102",
            "domain": "waves",
            "topics": [
                "wave_equation",
                "doppler_effect",
                "sound_intensity",
                "decibels",
                "standing_waves",
                "interference",
                "superposition",
                "harmonics"
            ],
            "input_types": ["text", "json"],
            "output_types": ["text", "analysis"],
            "version": "1.0.0",
            "framework": "strands"
        }

    def _get_description(self) -> str:
        return "Waves agent for Physics 102: handles wave mechanics, Doppler effect, sound, and interference problems"
