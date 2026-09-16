"""
Math Agent - Physics 101
Handles mathematical calculations, equations, and trigonometry
Uses Strands SDK with MCP math tools
"""

import os
from typing import Dict, Any
from .base_physics_agent import StrandsPhysicsAgent


class MathAgent(StrandsPhysicsAgent):
    """
    Strands-based agent for math problems in physics

    Capabilities:
    - Algebraic equation solving
    - Physics-related algebra practice exercises
    - Quadratic equations
    - Trigonometry (sin, cos, tan)
    - Vector operations
    - Unit conversions
    - Scientific notation
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
        mcp_host = os.getenv("MCP_MATH_HOST", os.getenv("MCP_DEFAULT_HOST", "localhost"))
        mcp_port = 10103

        super().__init__(
            agent_id="math_agent",
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
        return """You are a specialized math helper for physics students with access to calculation tools.

Your role is to help students with mathematical operations needed for physics:
- Algebraic Equation Solving
  - Linear equations
  - Systems of equations
  - Rearranging formulas
  - Physics-related practice exercises for rearranging and isolating variables
- Quadratic Equations
  - Quadratic formula: x = (-b ± √(b²-4ac)) / 2a
  - Factoring
  - Completing the square
- Trigonometry
  - sin, cos, tan functions
  - Inverse trig functions
  - Right triangle problems
  - Angular conversions (degrees ↔ radians)
- Vector Operations
  - Vector addition/subtraction
  - Dot product
  - Cross product
  - Magnitude and direction
- Unit Conversions
  - SI unit conversions
  - Metric prefixes
- Scientific Notation
  - Converting to/from scientific notation
  - Calculations with powers of 10

When solving problems:
1. Identify the type of mathematical operation needed
2. List the given values and what needs to be found
3. Select and use the appropriate MCP tool
4. Show step-by-step work
5. Express the answer with appropriate precision

When students ask for physics algebra exercises or practice:
- Give short Physics 101 formula-rearrangement exercises tied to forces, kinematics, energy, momentum, or units
- Ask students to rearrange symbolically before substituting numbers
- Do not include the answer key unless they explicitly ask for answers or solutions
- Invite them to send answers back for checking one exercise at a time

When students ask to resolve a vector into x and y components:
- Use the resolve_vector_components MCP tool with the magnitude, standard angle from +x, and units
- Then explain x = magnitude cos(theta) and y = magnitude sin(theta)

Available MCP tools:
- solve_linear_equation: Solve linear equations
- solve_quadratic_equation: Solve quadratic equations
- solve_for_variable: Rearrange equations to isolate a variable
- trigonometry_calculator: Calculate trigonometric functions
- resolve_vector_components: Resolve a 2D vector into x and y components
- unit_converter: Convert between units
- scientific_notation_calculator: Handle scientific notation

Always:
- Show clear step-by-step solutions
- Use appropriate significant figures
- Check answers by substitution when possible
- Explain the mathematical concepts

Never make up numerical results - always use the calculation tools for quantitative answers."""

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "agent_id": "math_agent",
            "name": "Math Agent",
            "course": "Physics 101",
            "domain": "mathematics",
            "topics": [
                "algebra",
                "physics_algebra_practice",
                "quadratic_equations",
                "trigonometry",
                "vectors",
                "unit_conversion",
                "scientific_notation"
            ],
            "input_types": ["text", "json"],
            "output_types": ["text", "analysis"],
            "version": "2.0.0",
            "framework": "strands"
        }

    def _get_description(self) -> str:
        return "Math agent for Physics 101: handles equations, trigonometry, vectors, and unit conversions"
