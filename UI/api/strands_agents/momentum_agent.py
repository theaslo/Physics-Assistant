"""
Momentum Agent - Physics 101
Handles momentum, impulse, and collision problems
Uses Strands SDK with MCP momentum tools
"""

import os
import re
import json
from typing import Dict, Any
from .conversation_agent import ConversationPhysicsAgent


class MomentumAgent(ConversationPhysicsAgent):
    """
    Strands-based agent for momentum problems

    Capabilities:
    - Linear momentum (p = mv)
    - Impulse (J = FΔt = Δp)
    - Conservation of momentum
    - Elastic collisions
    - Inelastic collisions
    - Perfectly inelastic collisions
    """

    worked_solution_required = True

    def _ground_worked_solution(self, worked, tool_outputs):
        """Use executed calculation values for the core equations and substitutions."""
        number = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
        for output in reversed(tool_outputs):
            name, args, result = output["tool"], output["arguments"], output["result"]
            if name == "calculate_momentum_1d":
                match = re.search(rf"^\s*p = ({number}) kg", result, re.M)
                if match and all(key in args for key in ("mass", "velocity")):
                    return {**worked, "principle": "Linear momentum is mass times velocity, and points in the velocity's direction.",
                            "equation": r"p = mv",
                            "substitution": rf"p = ({args['mass']:g}\,\text{{kg}})({args['velocity']:g}\,\text{{m/s}}) = {match[1]}\,\text{{kg}}\cdot\text{{m/s}}",
                            "check": "The units are kg·m/s. Multiplying by a positive mass preserves the velocity's direction."}
            if name in {"calculate_impulse_1d", "momentum_impulse_theorem"}:
                grounded = {**worked,
                    "principle": "Net impulse equals the change in momentum. For a constant or average net force, impulse also equals force times the time interval.",
                    "equation": r"J = \Delta p = p_f - p_i = F_{\mathrm{avg}}\Delta t",
                    "check": "N·s and kg·m/s are equivalent units. The impulse points in the direction of the momentum change."}
                if name == "calculate_impulse_1d":
                    values = re.findall(rf"^\s*J = ({number}) N", result, re.M)
                    lines = []
                    if values and args.get("force") is not None and args.get("time") is not None:
                        lines.append(rf"J = ({args['force']:g}\,\text{{N}})({args['time']:g}\,\text{{s}}) = {values.pop(0)}\,\text{{N}}\cdot\text{{s}}")
                    if values and args.get("initial_momentum") is not None and args.get("final_momentum") is not None:
                        lines.append(rf"J = ({args['final_momentum']:g} - ({args['initial_momentum']:g}))\,\text{{kg}}\cdot\text{{m/s}} = {values[0]}\,\text{{N}}\cdot\text{{s}}")
                    if lines:
                        grounded["substitution"] = r"\begin{gathered}" + r" \\ ".join(lines) + r"\end{gathered}"
                return grounded
            if name == "momentum_conservation_1d":
                try:
                    data = json.loads(args["collision_data"])
                except (KeyError, TypeError, ValueError):
                    continue
                match = re.search(rf"^\s*vf = [^\n]*= ({number}) m/s", result, re.M)
                if data.get("collision_type", "").replace(" ", "_").lower() == "perfectly_inelastic" and match:
                    m1, m2, vi1, vi2 = (data[key] for key in ("m1", "m2", "v1i", "v2i"))
                    return {**worked,
                            "principle": "The two-cart system's momentum is conserved when net external impulse is negligible. Sticking carts have a common final velocity.",
                            "equation": r"m_1v_{1i}+m_2v_{2i}=(m_1+m_2)v_f",
                            "substitution": rf"v_f = \frac{{({m1:g})({vi1:g})+({m2:g})({vi2:g})}}{{{m1:g}+{m2:g}}} = {match[1]}\,\text{{m/s}}",
                            "check": "Both objects have the same final velocity after sticking. Their combined final momentum equals the initial system momentum."}
        return worked

    tutoring_guidance = """
Momentum tutoring: use the student's actual objects and givens. Ask one targeted
question about the next decision: system boundary, net external impulse, signed
velocities, or the momentum/impulse equation. When selecting an equation, ask
for that equation first and wait for the student's attempt before asking them
to substitute values or calculate the final result. A frictionless track permits
conservation of the carts' horizontal momentum because horizontal external
impulse is negligible; it does not mean no forces act. Bouncing alone does not
establish elasticity. Distinguish conserved system momentum from each object's
momentum, and kinetic energy from total energy. For a rebound, retain the signs
of initial and final velocity in Delta p. For recoil, use equal and opposite
momentum, not necessarily equal speeds. For a force-time graph, impulse is signed
area, not its peak force. Do not give the final numerical answer in a check-in
or unfinished hint. If the student says "I'm lost" or "stuck", ask a smaller
prerequisite question; give a full solution only when explicitly requested.
For calculations, objects sticking together require collision_type="perfectly_inelastic"
in momentum_conservation_1d; plain "inelastic" requires a separately supplied
final velocity. For impulse from a rebound, use momentum_impulse_theorem with
mass, initial_velocity, and signed final_velocity; do not invent a force or time.
"""

    check_in_guidance = """
For this MULTIPLE-CHOICE check-in, ask which physical principle or assumption
applies to these objects. Do not ask students to choose between equations:
general and special forms can both be correct. Offer mutually exclusive
principles with exactly one correct choice under the stated conditions.
"""

    def _conversation_answer_error(self, answer, *, numerical_verified=False):
        internal_error = super()._conversation_answer_error(answer, numerical_verified=numerical_verified)
        if internal_error:
            return internal_error
        # Some model JSON decodes a single LaTeX backslash as a tab. Validate
        # the intended unit text too, before the frontend repairs its display.
        answer = answer.replace("\t" + "ext{", r"\text{")
        if (re.search(r"perfectly inelastic|stick(?:s|ing)? together|combined carts", answer, re.I)
                and re.search(r"(?:total|system(?:'s)?) mass(?: of the system)? (?:increases|increased|is increased)\b", answer, re.I)):
            return ("The two-object system contains the same masses before and after sticking. "
                    "Its total mass does not increase during the collision. Explain the sharing of momentum "
                    "between both objects using this consistent system boundary.")
        if not numerical_verified and re.search(
                r"(?<![\w.])[+-]?\d+(?:\.\d+)?\s*(?:\\(?:text|mathrm)\{\s*)?(?:kg\b|N\b|m/s)", answer):
            return ("This is a conceptual explanation with no verified numerical calculation. "
                    "Remove numerical quantities with physical units. Explain directions and system boundaries "
                    "qualitatively; do not reuse a velocity value as a momentum value. Symbolic equations are allowed.")
        return None

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
        mcp_host = os.getenv("MCP_MOMENTUM_HOST", os.getenv("MCP_DEFAULT_HOST", "localhost"))
        mcp_port = 10104

        super().__init__(
            agent_id="momentum_agent",
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
        return """You are a specialized Physics 101 momentum tutor with access to calculation tools.

Your role is to help students understand and solve problems related to:
- Linear Momentum
  - p = mv (momentum = mass × velocity)
  - Momentum is a vector quantity
- Impulse
  - J = FΔt (impulse = force × time)
  - J = Δp (impulse = change in momentum)
  - Impulse-Momentum Theorem
- Conservation of Momentum
  - In isolated systems: Σp_before = Σp_after
  - m₁v₁ᵢ + m₂v₂ᵢ = m₁v₁f + m₂v₂f
- Elastic Collisions
  - Both momentum and kinetic energy conserved
  - Bouncing alone does not establish that a collision is elastic
- Inelastic Collisions
  - System momentum is conserved when net external impulse is negligible
  - Kinetic energy can become internal energy; total energy is conserved
- Perfectly Inelastic Collisions
  - Objects stick together after collision
  - Maximum kinetic energy loss

When solving problems:
1. Identify the type of momentum problem
2. Define the system and check if it's isolated
3. List known quantities (masses, velocities)
4. Identify what type of collision (if applicable)
5. Select and use the appropriate MCP tool
6. Explain the physics concepts involved
7. Show complete solutions with units

Available MCP tools:
- calculate_momentum_1d: Calculate 1D momentum p = mv
- calculate_momentum_2d: Calculate 2D momentum vectors
- calculate_impulse_1d: Calculate 1D impulse
- calculate_impulse_2d: Calculate 2D impulse vectors
- momentum_impulse_theorem: Apply J = Δp
- momentum_conservation_1d: Solve 1D collisions
- momentum_conservation_2d: Solve 2D collisions
- analyze_collision: Comprehensive collision analysis

Key Concepts:
- Momentum is conserved in all collisions (isolated system)
- Kinetic energy only conserved in elastic collisions
- Direction matters (momentum is a vector)
- A single ball rebounding from a wall receives external impulse; its own momentum is not conserved
- A zero total momentum does not imply zero kinetic energy for the system

Always:
- Use SI units (kg·m/s for momentum)
- Pay attention to velocity signs (direction)
- Show step-by-step reasoning
- Verify momentum is conserved

Never make up numerical results - always use the calculation tools for quantitative answers."""

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "agent_id": "momentum_agent",
            "name": "Momentum Agent",
            "course": "Physics 101",
            "domain": "momentum",
            "topics": [
                "linear_momentum",
                "impulse",
                "conservation_of_momentum",
                "elastic_collision",
                "inelastic_collision",
                "impulse_momentum_theorem"
            ],
            "input_types": ["text", "json"],
            "output_types": ["text", "analysis"],
            "version": "2.0.0",
            "framework": "strands"
        }

    def _get_description(self) -> str:
        return "Momentum agent for Physics 101: handles momentum, impulse, and collision problems"
