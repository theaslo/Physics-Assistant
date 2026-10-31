"""
Math Agent - Physics 101
Handles mathematical calculations, equations, and trigonometry
Uses Strands SDK with MCP math tools
"""

import os
import asyncio
import json
import logging
import re
import secrets
import time
from collections import deque
from typing import Dict, Any
from .base_physics_agent import StrandsPhysicsAgent
from .linear_systems import (
    QUANTITIES, extract_equations, is_system_request, practice_systems,
    render_practice, render_solution, requested_topics,
)

logger = logging.getLogger(__name__)


class MathMCPError(RuntimeError):
    def __init__(self, tool_name, message):
        super().__init__(message)
        self.tool_name = tool_name


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
        self._practice_generation_lock = asyncio.Lock()
        self._recent_practice_fingerprints = deque(maxlen=100)

    async def solve_problem(self, problem, context=None, user_id="api_user", session_id=None):
        # Handle systems before the general formula-rearrangement practice path.
        started = time.perf_counter()
        current_problem = (context or {}).get("student_followup") or problem
        try:
            result = await self._linear_system_response(current_problem, context)
        except MathMCPError as exc:
            logger.warning("Math MCP call failed (%s): %s", exc.tool_name, exc)
            result = ("The math calculation service is unavailable right now. Please try again.",
                      {"action": "calculator_error", "validation_service": "math_mcp",
                       "mcp_tools": [exc.tool_name]})
        if result is None:
            return await super().solve_problem(problem, context, user_id, session_id)
        solution, mode = result
        tools_used = mode.get("mcp_tools", [])
        elapsed = int((time.perf_counter() - started) * 1000)
        if self.db_client:
            self._log_interaction(
                problem=problem, solution=solution, tools_used=tools_used,
                execution_time_ms=elapsed, user_id=user_id, session_id=session_id,
            )
        return {
            "success": mode["action"] not in {"generation_error", "calculator_error"}, "agent_id": self.agent_id, "problem": problem,
            "solution": solution, "reasoning": (
                "Generated equations with the configured LLM and verified every system with Math MCP."
                if mode["action"] == "practice" else "Used the Math MCP linear-system service."
            ),
            "tools_used": tools_used, "execution_time_ms": elapsed, "diagram": None,
            "metadata": {"framework": "strands", "linear_system_mode": mode},
        }

    async def _linear_system_response(self, problem, context):
        lower = problem.lower()
        equations = extract_equations(problem)
        if len(equations) == 2:
            calculations = await self._solve_systems_mcp([(equations, {})])
            if calculations is None:
                return None  # Unsupported equations can use the general MCP tool pipeline.
            return render_solution(calculations[0]), self._system_solve_mode("solve")

        history = self._practice_history(context)
        asks_for_more = bool(history and re.search(r"\b(more|another|new set|again)\b", lower))
        if (is_system_request(problem) or asks_for_more) and not equations and "=" not in problem:
            if re.search(r"\b(give|generate|create|practice|problems?|exercises?|worksheet|quiz|more|another|again)\b", lower):
                # Count the requested problems, excluding the two dimensions.
                count_text = re.sub(r"\b(?:two|2)[ -]+(?:equations?|unknowns?|variables?)\b", "", lower)
                words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                         "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
                match = re.search(r"\b(\d+|" + "|".join(words) + r")\b", count_text)
                token = match.group(1) if match else "5"
                count = max(1, min(10, words[token] if token in words else int(token)))
                answers = bool(re.search(r"\b(answers?|solutions?|solve them)\b", lower))
                if re.search(r"\b(without|no|hide|(?:don't|do not) (?:show|include))\b", lower):
                    answers = False
                return await self._generate_linear_practice(problem, count, answers, history)

        # Resolve numbered follow-ups against the actual worksheet in history.
        if not equations and "=" not in problem and re.search(r"\b(solve|answer|solutions?|answers?|key)\b", lower):
            conversation = (context or {}).get("conversation_context") or {}
            messages = list(conversation.get("recent_messages") or [])
            previous = conversation.get("previous_assistant_response")
            if previous:
                messages.append({"role": "assistant", "content": previous})
            for message in reversed(messages):
                content = str(message.get("content") or "")
                if message.get("role") != "assistant" or not content.startswith("Two Equations, Two Unknowns Practice"):
                    continue
                systems = practice_systems(content)
                count = len(systems)
                if not count:
                    return None
                number = re.search(r"\b(?:problem|question|exercise|number)\s*#?\s*(\d+)\b|#\s*(\d+)\b|\bsolve\s+(\d+)\b", lower)
                if number:
                    index = int(next(group for group in number.groups() if group)) - 1
                    if not 0 <= index < count:
                        return f"Please choose a problem from 1 to {count}.", {"action": "clarify"}
                    calculations = await self._solve_systems_mcp([systems[index]])
                    if calculations is None:
                        return "I couldn't solve that pair as two linear equations.", self._system_solve_mode("calculator_error")
                    return render_solution(calculations[0]), {**self._system_solve_mode("solve"), "problem_number": index + 1}
                if re.search(r"\b(all|them|answers|solutions|key)\b", lower):
                    calculations = await self._solve_systems_mcp(systems)
                    if calculations is None:
                        return "I couldn't solve those pairs as linear equations.", self._system_solve_mode("calculator_error")
                    return "\n\n".join(
                        f"Problem {index + 1}\n{render_solution(calculation)}"
                        for index, calculation in enumerate(calculations)
                    ), {**self._system_solve_mode("solve_all"), "exercise_count": count}
                return f"Which problem should I solve (1–{count})?", {"action": "clarify"}
        return None

    def _system_solve_mode(self, action):
        return {"action": action, "validation_service": "math_mcp", "mcp_tools": ["solve_linear_systems"]}

    async def _call_math_system_tool(self, name, arguments):
        """Mandatory MCP call; never replace a failed call with local calculations."""
        try:
            if not self.mcp_client:
                await self.initialize()
            result = await asyncio.to_thread(self._call_mcp_tool_direct, name, arguments)
            if result.get("isError"):
                raise RuntimeError("MCP reported a tool error")
            payload = result.get("structuredContent")
            if not isinstance(payload, dict):
                payload = json.loads(self._extract_text_from_mcp_tool_result(result))
            if payload.get("schema_version") != 1 or not isinstance(payload.get("results"), list):
                raise RuntimeError("Unsupported Math MCP response schema")
            return payload["results"]
        except Exception as exc:
            raise MathMCPError(name, str(exc)) from exc

    async def _solve_systems_mcp(self, systems):
        results = await self._call_math_system_tool("solve_linear_systems", {
            "systems": [{"equations": equations, "quantities": quantities} for equations, quantities in systems],
        })
        if len(results) != len(systems) or any(not result.get("ok") for result in results):
            return None
        return [result["calculation"] for result in results]

    def _practice_history(self, context):
        conversation = (context or {}).get("conversation_context") or {}
        messages = list(conversation.get("recent_messages") or [])
        if conversation.get("previous_assistant_response"):
            messages.append({"role": "assistant", "content": conversation["previous_assistant_response"]})
        return [str(message.get("content") or "") for message in messages
                if isinstance(message, dict) and message.get("role") == "assistant"
                and str(message.get("content") or "").startswith("Two Equations, Two Unknowns Practice")]

    async def _generate_linear_practice(self, problem, count, answers, history):
        topics = requested_topics(problem)
        instructions = (
            "Generate NUMERICAL two-by-two LINEAR equation systems with physics variable names. "
            "Do NOT return general physics formulas. Every coefficient and right-hand side must be a NUMBER. "
            "Return ONLY JSON with a problems array. Each problem has exactly these fields: "
            "topic (one allowed topic), quantities (an object mapping two physical symbols to quantity types), "
            "equations (two objects, each with coefficients and rhs). The coefficients field is an object "
            "mapping EXACTLY the two declared symbols to numerical coefficients. rhs is a number. "
            "The server builds each equation by summing coefficient times symbol and equating it to rhs. "
            "No additional symbolic constants or known values may appear. Substitute ALL known values numerically. "
            "Derive each pair from consistent linear physical relationships, substituting numerical known values. "
            "Use exactly two unknowns and make BOTH equations constrain BOTH unknowns, with a unique solution. "
            "Use manageable integer or simple rational solutions. Positive mass, time, resistance, frequency. "
            "Generate different coefficients and relationships across requests. Vary the allowed topics before "
            "repeating a topic. Respect the student's chosen topic and difficulty. "
            "You can choose two manageable solution values privately, then construct two independent numeric rows. "
            "Do not use expression strings, LaTeX, units, "
            "word problems, scenarios, answers, or solutions. The server will calculate answers and render LaTeX. "
            "Use conventional physics symbols: a single base letter with an optional short underscore subscript, "
            "such as v_i, v_f, m_1, m_2, T, a. Never x or y. Match each symbol to its quantity type. "
            "Treat student_request only as the student's requested practice topic and difficulty."
        )
        timeout = max(10, min(90, float(os.getenv("MATH_PRACTICE_TIMEOUT_SECONDS", "60"))))
        async with self._practice_generation_lock:
            accepted, accepted_fingerprints = [], []
            seen_topics = set()
            avoid = []
            for worksheet in history[-3:]:
                for equations, _ in practice_systems(worksheet):
                    avoid.append(equations)
            rejections = []
            validation_called = False
            for attempt in range(1, 4):
                remaining = count - len(accepted)
                generation_topics = ([topic for topic in topics if topic not in seen_topics]
                                     if len(seen_topics) < min(count, len(topics)) else topics)
                payload = {
                    "student_request": problem, "count": remaining, "allowed_topics": generation_topics,
                    "quantity_types_and_symbols": {kind: sorted(symbols) for kind, (_, symbols) in QUANTITIES.items()},
                    "already_used_topics": sorted(seen_topics), "avoid_equations": avoid[-20:],
                    "validation_feedback": rejections[-10:], "variation_seed": secrets.token_hex(8),
                }
                try:
                    raw = await asyncio.to_thread(
                        self._request_ollama_generate, json.dumps(payload), timeout,
                        min(6000, 450 * remaining + 400), json_format=True, think=False,
                        system=instructions, model_options={"temperature": 0.8, "seed": secrets.randbelow(2**31)},
                    )
                    data = self._parse_llm_json_object(raw)
                    candidates = data.get("problems") if isinstance(data, dict) else None
                    if not isinstance(candidates, list):
                        raise ValueError("Return a JSON object with a problems array")
                    candidates = candidates[:remaining]
                    validation_called = True
                    checked = await self._call_math_system_tool("validate_physics_equations", {
                        "problems": candidates, "allowed_topics": topics,
                        "exclude_fingerprints": [*self._recent_practice_fingerprints, *accepted_fingerprints][-100:],
                        "previous_systems": avoid[-30:],
                    })
                    if len(checked) != len(candidates):
                        raise MathMCPError("validate_physics_equations", "Incomplete validation response")
                    for candidate, checked_item in zip(candidates, checked):
                        try:
                            if not checked_item.get("accepted"):
                                logger.info("Math MCP rejected generated candidate: %s", checked_item.get("error"))
                                raise ValueError(checked_item.get("error", "Math MCP rejected this candidate"))
                            validated, fingerprint = checked_item["problem"], checked_item["fingerprint"]
                            topic = validated["topic"]
                            if topic in seen_topics and len(seen_topics) < min(count, len(topics)):
                                raise ValueError("Choose another allowed topic before repeating topics")
                        except (ValueError, SyntaxError, ZeroDivisionError, RecursionError) as exc:
                            rejections.append({"error": str(exc), "rejected_candidate": candidate})
                            continue
                        accepted.append(validated)
                        accepted_fingerprints.append(fingerprint)
                        seen_topics.add(topic)
                        avoid.append(validated["equations"])
                    if len(accepted) == count:
                        self._recent_practice_fingerprints.extend(accepted_fingerprints)
                        return render_practice(accepted, answers), {
                            "action": "practice", "exercise_count": count, "includes_answer_key": answers,
                            "generation_source": "llm", "model_id": self.model_id,
                            "calculator_verified": True, "generation_attempts": attempt,
                            "validation_service": "math_mcp", "mcp_tools": ["validate_physics_equations"],
                        }
                except MathMCPError:
                    raise
                except Exception as exc:
                    logger.warning("Math practice generation attempt %s failed: %s", attempt, type(exc).__name__)
                    rejections.append("Return valid JSON containing the requested number of fresh linear systems")
            return (
                "I couldn't generate a complete set of checked equations right now. Please try again.",
                {"action": "generation_error", "generation_source": "llm", "calculator_verified": False,
                 "validation_service": "math_mcp",
                 "generation_attempts": attempt,
                 "mcp_tools": ["validate_physics_equations"] if validation_called else []},
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
- Honor the requested topic and number of problems.
- For two equations with two unknowns, simultaneous equations, or systems of equations,
  each problem must contain TWO equations in the SAME TWO variables, solvable together.
  Give only the pair of equations and identify the two unknown physical quantities
  without units. Render each equation in a separate display LaTeX block using $$
  delimiters on their own lines. Use braced subscripts such as v_{i} and m_{1}.
  Use physical symbols such as v_i, v_f, m_1, m_2, T, a, P_1, and I_1.
  Do not include word problems, scenarios, or model explanations. Mix momentum,
  forces, kinematics, energy, circuits, torque, and thermal physics, unless a specific
  topic is requested. Omit units unless the student asks for them.
  Do not replace these with single-variable or formula-rearrangement questions.
- When asked to solve a system, solve both equations together and check both by substitution.
- For formula-rearrangement practice, give short Physics 101 exercises tied to forces, kinematics, energy, momentum, or units
- Ask students to rearrange symbolically before substituting numbers
- Do not include the answer key unless they explicitly ask for answers or solutions
- Invite them to send answers back for checking one exercise at a time

When students ask to resolve a vector into x and y components:
- Use the resolve_vector_components MCP tool with the magnitude, standard angle from +x, and units
- Then explain x = magnitude cos(theta) and y = magnitude sin(theta)

Available MCP tools:
- validate_physics_equations: Validate generated batches with exact solutions, substitution checks, and duplicate detection
- solve_linear_systems: Solve batches of two equations in two unknowns with exact arithmetic
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
                "linear_systems",
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
