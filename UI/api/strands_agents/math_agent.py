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

Available MCP tools:
- solve_equation: Solve algebraic equations
- quadratic_solver: Solve quadratic equations
- trig_functions: Trigonometric calculations
- vector_operations: Vector math operations
- unit_converter: Convert between units
- scientific_notation: Handle scientific notation

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
