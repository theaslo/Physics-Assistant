"""
Forces Agent - Physics 101
Handles Newton's laws, springs, friction, and equilibrium problems
Uses Strands SDK with MCP forces tools
"""

import os
import asyncio
import json
from typing import Dict, Any, Optional
from .base_physics_agent import StrandsPhysicsAgent


class ForcesAgent(StrandsPhysicsAgent):
    """
    Strands-based agent for forces problems

    Capabilities:
    - Newton's Second Law (F=ma)
    - Spring force (Hooke's Law)
    - Friction (static and kinetic)
    - Force components and vectors
    - Equilibrium problems
    - Inclined plane problems
    - Tension and pulley systems
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
        mcp_host = os.getenv("MCP_FORCES_HOST", os.getenv("MCP_DEFAULT_HOST", "localhost"))
        mcp_port = 10100

        super().__init__(
            agent_id="forces_agent",
            mcp_host=mcp_host,
            mcp_port=mcp_port,
            llm_host=llm_host,
            model_id=model_id,
            database_api_url=database_api_url,
            enable_database_logging=enable_database_logging,
            enable_rag=enable_rag,
            rag_api_url=rag_api_url
        )

    async def _solve_conversation_turn(
        self, problem: str, context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Let the model select the active problem without keyword-based answer fallbacks."""
        question = self._current_followup_text(problem, context)
        history = self._conversation_context_text(problem, context, include_current=False)
        if not history and "Previous conversation:" in problem:
            history = problem.split("Previous conversation:", 1)[1].split("Current follow-up question:", 1)[0]

        catalog = self._fast_mcp_tool_catalog()
        tools_used = []
        tool_outputs = []
        diagrams = []

        async def call_tool(name: str, arguments: Dict[str, Any]) -> None:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._call_mcp_tool_direct, name, arguments),
                timeout=self.fast_mcp_explain_timeout_seconds,
            )
            output = self._extract_text_from_mcp_tool_result(result)
            if not output or self._is_mcp_error_text(output) or result.get("status") == "error":
                raise RuntimeError(f"The {name} tool could not verify this question. Please try again.")
            tools_used.append(name)
            tool_outputs.append({"tool": name, "arguments": arguments, "result": self._strip_embedded_diagram_json(output)})
            diagram = self._extract_diagram_from_text(output)
            if diagram:
                diagrams.append(diagram)

        # Conceptual questions need physics grounding without invented numerical inputs.
        await call_tool("get_force_principles", {})
        prompt = (
            "You are a conversational physics tutor. Interpret the CURRENT QUESTION using the history only when relevant.\n"
            "The history may contain several unrelated problems. Decide which situation the student means now. "
            "A new object or situation starts a new problem, even in a short or misspelled question. "
            "Do not transfer masses, spring constants, angles, coefficients, or motion directions from a different problem. "
            "For a genuine follow-up, preserve the relevant setup; apply any changed conditions the student gives. "
            "Explicit references to an earlier problem can return to that problem. If ambiguous, ask one specific question.\n"
            "Answer the current question directly and briefly; do not repeat a full previous solution. "
            "Say yes or no only when the student proposes a claim, and ensure it agrees with your explanation. "
            "For a direction question, do not require numbers that cannot affect the direction. "
            "If motion direction is unknown, explain the conditional directions and ask which way it is moving. "
            "For quiz reflection or hints, respect the student's requested pace and ask at most one question.\n"
            "Use MCP principles for conceptual answers and MCP calculation tools for numerical results. "
            "Never invent inputs to satisfy a tool requirement. Tool results below apply ONLY to their stated arguments. "
            "Do not mention tools or implementation details in the student answer.\n"
            "Return JSON with exactly these fields: "
            '{"answer":"student-facing answer, or empty if calculations are needed",'
            '"tool_calls":[{"tool_name":"catalog name","arguments":{}}],"show_diagram":false}. '
            "If you need a calculation, request up to three tools and wait for their results before answering. "
            "Request dependent calculations in separate rounds. If no calculation is needed, answer using the principles. "
            "Set show_diagram=true only if the student requests a diagram and a relevant tool produced one.\n\n"
            f"Tool catalog: {json.dumps(catalog, ensure_ascii=True)}\n\n"
            f"Conversation history (oldest first):\n{history}\n\n"
            f"CURRENT QUESTION:\n{question}\n"
        )
        for _ in range(3):
            try:
                response = await asyncio.to_thread(
                    self._request_ollama_generate,
                    prompt + "\nVerified MCP results:\n" + json.dumps(tool_outputs, ensure_ascii=True),
                    self.fast_mcp_explain_timeout_seconds,
                    900,
                    True,
                    self.model_id,
                    think=False if self.model_id.startswith("qwen3") else None,
                )
            except Exception as exc:
                raise RuntimeError(
                    "The tutor could not reach the language model to interpret your current question. Please try again."
                ) from exc
            parsed = self._parse_llm_json_object(response)
            if not parsed:
                raise RuntimeError("The tutor could not interpret this question. Please try again.")
            calls = self._extract_fast_mcp_tool_calls(parsed, catalog)
            if calls:
                for call in calls:
                    await call_tool(call["tool_name"], call["arguments"])
                continue
            answer = parsed.get("answer")
            if parsed.get("tool_calls") or not isinstance(answer, str) or not answer.strip():
                raise RuntimeError("The tutor could not verify an answer to this question. Please try again.")
            return {
                "solution": self._strip_embedded_diagram_json(answer).strip(),
                "tools_used": list(dict.fromkeys(tools_used)),
                "diagram": diagrams[-1] if diagrams and parsed.get("show_diagram") is True else None,
            }
        raise RuntimeError("This calculation needs more steps than the tutor could complete. Please try a smaller step.")

    def _get_system_prompt(self) -> str:
        return """You are a specialized Physics 101 forces tutor with access to calculation tools.

Your role is to help students understand and solve problems related to:
- Newton's Laws of Motion
  - First Law (Inertia)
  - Second Law (F = ma)
  - Third Law (Action-Reaction)
- Spring Forces (Hooke's Law: F = -kx)
- Friction Forces
  - Static friction (f_s ≤ μ_s N)
  - Kinetic friction (f_k = μ_k N)
- Force Components and Vectors
- Equilibrium (ΣF = 0)
- Inclined Plane Problems
- Tension and Pulley Systems

When solving problems:
1. Identify all forces acting on the object(s)
2. Draw or describe a free body diagram
3. Choose an appropriate coordinate system
4. List known quantities with proper units
5. Identify the unknown quantity to find
6. Select and use the appropriate MCP tool for calculations
7. Explain the physics concepts involved
8. Show complete solutions with units

When a student asks to analyze a mistake from a Newton's First Law or Second Law quiz:
- Treat it as reflection on their existing quiz, not as a brand-new generic force problem
- Do not invent a new multiple-choice check
- Ask for the quiz question, their answer, the correct answer, and why they chose their answer if any of those are missing
- Focus feedback on the misconception: motion vs changing motion for First Law, and net external force vs individual forces for Second Law

Available MCP tools:
- newton_second_law: Calculate force, mass, or acceleration using F=ma
- calculate_spring_force_tool: Calculate spring force using Hooke's Law
- calculate_friction_force_tool: Calculate static or kinetic friction
- resolve_force_components: Resolve forces into components
- check_equilibrium: Analyze forces in equilibrium
- analyze_forces_on_incline: Solve inclined plane problems
- analyze_tension_forces: Solve rope/string tension and pulley systems
- create_free_body_diagram: Generate free-body force breakdowns
- add_forces_2d: Add multiple 2D force vectors

Always:
- Use SI units (N, kg, m/s²)
- Show step-by-step reasoning
- Draw attention to common misconceptions
- Verify answers make physical sense

Never make up numerical results - always use the calculation tools for quantitative answers."""

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "agent_id": "forces_agent",
            "name": "Forces Agent",
            "course": "Physics 101",
            "domain": "forces",
            "topics": [
                "newtons_laws",
                "spring_force",
                "friction",
                "force_components",
                "equilibrium",
                "inclined_plane",
                "tension",
                "pulleys"
            ],
            "input_types": ["text", "json"],
            "output_types": ["text", "analysis"],
            "version": "2.0.0",
            "framework": "strands"
        }

    def _get_description(self) -> str:
        return "Forces agent for Physics 101: handles Newton's laws, springs, friction, and equilibrium problems"
