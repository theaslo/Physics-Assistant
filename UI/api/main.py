"""
FastAPI server for Physics Assistant API
All physics agents now use Strands SDK with MCP tools and Ollama LLM
"""

import asyncio
import logging
import math
import re
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os

from hitl_knowledge_transfer import KnowledgeTransferGate

# Import all Strands-based agents
from strands_agents import (
    # Physics 101
    ForcesAgent,
    KinematicsAgent,
    MathAgent,
    MomentumAgent,
    EnergyAgent,
    AngularMotionAgent,
    # Physics 102
    ThermodynamicsAgent,
    WavesAgent,
    # Physics 201
    ElectromagnetismAgent,
    # Physics 202
    OpticsAgent,
    ModernPhysicsAgent,
    # Base class for type hints
    StrandsPhysicsAgent
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Valid agent IDs by course level - ALL use Strands SDK now
PHYSICS_101_AGENTS = ["forces_agent", "kinematics_agent", "math_agent", "momentum_agent", "energy_agent", "angular_motion_agent"]
PHYSICS_102_AGENTS = ["thermodynamics_agent", "waves_agent"]
PHYSICS_201_AGENTS = ["electromagnetism_agent"]
PHYSICS_202_AGENTS = ["optics_agent", "modern_physics_agent"]
ALL_VALID_AGENTS = PHYSICS_101_AGENTS + PHYSICS_102_AGENTS + PHYSICS_201_AGENTS + PHYSICS_202_AGENTS

# Global agent store for managing active agents (all Strands-based now)
agent_store: Dict[str, StrandsPhysicsAgent] = {}
knowledge_transfer_gate: Optional[KnowledgeTransferGate] = None

# Pydantic models for API requests/responses
class AgentCreateRequest(BaseModel):
    """Request model for creating a physics agent"""
    agent_id: str = Field(
        ...,
        description="Agent type: Physics 101 (forces, kinematics, math, momentum, energy, angular_motion) or Physics 102-202 (thermodynamics, waves, electromagnetism, optics, modern_physics)",
        pattern="^(forces_agent|kinematics_agent|math_agent|momentum_agent|energy_agent|angular_motion_agent|thermodynamics_agent|waves_agent|electromagnetism_agent|optics_agent|modern_physics_agent)$"
    )
    use_direct_tools: bool = Field(
        default=True, 
        description="Whether to use direct MCP tools (recommended)"
    )
    enable_rag: bool = Field(
        default=True,
        description="Whether to enable RAG context augmentation"
    )
    rag_api_url: Optional[str] = Field(
        default=None,
        description="URL for RAG API server (defaults to DATABASE_API_HOST env var)"
    )

class AgentCreateResponse(BaseModel):
    """Response model for agent creation"""
    success: bool
    agent_id: str
    message: str
    capabilities: Optional[Dict[str, Any]] = None

class ProblemSolveRequest(BaseModel):
    """Request model for solving physics problems"""
    problem: str = Field(..., description="Physics problem description")
    context: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Optional context for the problem"
    )
    user_id: Optional[str] = Field(
        default="api_user",
        description="User identifier for database logging"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session identifier for database logging"
    )

class ProblemSolveResponse(BaseModel):
    """Response model for problem solving"""
    success: bool
    agent_id: str
    problem: str
    solution: Optional[str] = None
    reasoning: Optional[str] = None
    tools_used: Optional[list] = None
    execution_time_ms: Optional[int] = None
    diagram: Optional[Dict[str, Any]] = None
    hitl: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class AgentHealthResponse(BaseModel):
    """Response model for agent health check"""
    agent_id: str
    status: str
    tools_count: int
    ready: bool
    mode: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global knowledge_transfer_gate
    logger.info("🚀 Starting Physics Assistant API")
    database_api_host = os.getenv("DATABASE_API_HOST", "localhost")
    database_api_port = os.getenv("DATABASE_API_PORT", "8001")
    database_api_url = f"http://{database_api_host}:{database_api_port}"
    knowledge_transfer_gate = KnowledgeTransferGate(
        database_api_url=database_api_url,
    )
    yield
    logger.info("🛑 Shutting down Physics Assistant API")
    if knowledge_transfer_gate:
        await knowledge_transfer_gate.cleanup()
        knowledge_transfer_gate = None
    # Cleanup agents if needed
    agent_store.clear()

# Create FastAPI application
app = FastAPI(
    title="Physics Assistant API",
    description="API for physics tutoring agents with MCP tool integration",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper functions
def _get_strands_agent_class(agent_id: str):
    """Get the Strands agent class for a given agent_id"""
    strands_agent_map = {
        # Physics 101
        "forces_agent": ForcesAgent,
        "kinematics_agent": KinematicsAgent,
        "math_agent": MathAgent,
        "momentum_agent": MomentumAgent,
        "energy_agent": EnergyAgent,
        "angular_motion_agent": AngularMotionAgent,
        # Physics 102
        "thermodynamics_agent": ThermodynamicsAgent,
        "waves_agent": WavesAgent,
        # Physics 201
        "electromagnetism_agent": ElectromagnetismAgent,
        # Physics 202
        "optics_agent": OpticsAgent,
        "modern_physics_agent": ModernPhysicsAgent,
    }
    return strands_agent_map.get(agent_id)


async def get_or_create_agent(agent_id: str, use_direct_tools: bool = True, enable_rag: bool = True, rag_api_url: str = None):
    """Get existing agent or create new one with RAG and database logging enabled

    All agents now use Strands SDK with MCP tools and Ollama LLM.
    The use_direct_tools parameter is kept for API compatibility but ignored.
    """
    import os
    database_api_host = os.getenv("DATABASE_API_HOST", "localhost")
    database_api_port = os.getenv("DATABASE_API_PORT", "8001")
    database_api_url = f"http://{database_api_host}:{database_api_port}"

    if rag_api_url is None:
        rag_api_url = database_api_url

    # Simplified key - all agents use Strands now
    agent_key = f"{agent_id}_{enable_rag}"

    if agent_key not in agent_store:
        logger.info(f"Creating new Strands agent: {agent_id} (RAG: {'enabled' if enable_rag else 'disabled'})")

        strands_class = _get_strands_agent_class(agent_id)

        if not strands_class:
            raise ValueError(f"Unknown agent_id: {agent_id}")

        agent = strands_class(
            database_api_url=database_api_url,
            enable_database_logging=True,
            enable_rag=enable_rag,
            rag_api_url=rag_api_url
        )

        await agent.initialize()
        agent_store[agent_key] = agent
        logger.info(f"Agent {agent_id} created (Strands SDK) with RAG: {'enabled' if enable_rag else 'disabled'}")

    return agent_store[agent_key]


def _append_perf_stage(
    stages: list[Dict[str, Any]],
    stage: str,
    started_at: float,
    **extra: Any,
) -> None:
    duration_ms = int((time.perf_counter() - started_at) * 1000)
    item: Dict[str, Any] = {"stage": stage, "duration_ms": duration_ms}
    item.update(extra)
    stages.append(item)


def _finalize_perf_trace(component: str, stages: list[Dict[str, Any]], started_at: float) -> Dict[str, Any]:
    total_ms = int((time.perf_counter() - started_at) * 1000)
    slowest = max(stages, key=lambda x: int(x.get("duration_ms", 0)), default={"stage": "none", "duration_ms": 0})
    return {
        "component": component,
        "total_ms": total_ms,
        "slowest_stage": slowest.get("stage", "none"),
        "slowest_duration_ms": int(slowest.get("duration_ms", 0)),
        "stages": stages,
    }


def _merge_performance_trace(
    api_trace: Dict[str, Any],
    hitl_trace: Optional[Dict[str, Any]] = None,
    agent_trace: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {"api": api_trace}
    if isinstance(hitl_trace, dict) and hitl_trace:
        merged["hitl"] = hitl_trace
    if isinstance(agent_trace, dict) and agent_trace:
        merged["agent"] = agent_trace

    candidates: list[Dict[str, Any]] = []
    for component_key in ("api", "hitl", "agent"):
        trace = merged.get(component_key)
        if not isinstance(trace, dict):
            continue
        for stage in trace.get("stages", []) or []:
            if not isinstance(stage, dict):
                continue
            candidates.append(
                {
                    "component": component_key,
                    "stage": str(stage.get("stage", "unknown")),
                    "duration_ms": int(stage.get("duration_ms", 0)),
                }
            )

    slowest = max(candidates, key=lambda x: int(x.get("duration_ms", 0)), default={"component": "none", "stage": "none", "duration_ms": 0})
    merged["request_total_ms"] = int(api_trace.get("total_ms", 0))
    merged["slowest_stage"] = slowest
    return merged


def _log_slow_trace(agent_id: str, user_id: Optional[str], performance_trace: Dict[str, Any]) -> None:
    total_ms = int(performance_trace.get("request_total_ms", 0))
    threshold_ms = int(os.getenv("PERF_SLOW_REQUEST_MS", "15000"))
    if total_ms < threshold_ms:
        return
    slowest = performance_trace.get("slowest_stage") or {}
    logger.warning(
        "SLOW_REQUEST_TRACE agent=%s user=%s total_ms=%s slowest_component=%s slowest_stage=%s slowest_ms=%s",
        agent_id,
        user_id or "unknown_user",
        total_ms,
        slowest.get("component", "unknown"),
        slowest.get("stage", "unknown"),
        int(slowest.get("duration_ms", 0)),
    )


def _concept_remediation_explanation(agent_id: str, concept_tag: str) -> str:
    if concept_tag == "hookes_law":
        return (
            "Hooke's law says the spring-force magnitude grows in direct proportion to stretch: |F_s| = k|x|. "
            "So putting k in the denominator reverses the meaning: a stiffer spring should make a larger force for the same stretch, not a smaller one."
        )
    if concept_tag in {"projectile_components", "projectile_peak", "range_formula"}:
        return (
            "Projectile motion separates into horizontal and vertical parts. The horizontal velocity stays constant when we ignore air resistance, "
            "while the vertical velocity changes because gravity acts downward."
        )
    if concept_tag == "newton_second_law":
        return (
            "Newton's second law connects the net force, not just one individual force, to acceleration: sum F = ma. "
            "The direction of the net force sets the direction of the acceleration."
        )
    if concept_tag == "incline_components":
        return (
            "For an incline, it helps to rotate the axes so one axis is parallel to the ramp and the other is perpendicular. "
            "Weight then splits into mg sin(theta) along the ramp and mg cos(theta) into the ramp."
        )
    if agent_id == "forces_agent":
        return "For force problems, start by identifying the object, drawing all forces on that object, and choosing clear axes."
    if agent_id == "kinematics_agent":
        return "For kinematics problems, start by listing known quantities, the unknown, and which direction you will call positive."
    return "Let's pause on the underlying concept before doing the algebra."


def _concept_next_steps(agent_id: str, concept_tag: str) -> list[str]:
    if concept_tag == "equation_selection":
        return [
            "Write the kinematics equation that matches the unknown and known values.",
            "Substitute the numerical values from the problem into that equation with units.",
            "Calculate the requested value and write the final answer with units.",
        ]
    if concept_tag == "hookes_law":
        return [
            "Write the magnitude form of Hooke's law: |F_s| = k|x|. Identify k and x from the problem before doing arithmetic.",
            "Substitute the given k and x values with units into |F_s| = k|x|, but pause before the final multiplication.",
            "After finding the force magnitude, decide the direction: a spring force points opposite the stretch or compression.",
        ]
    if concept_tag == "projectile_components":
        return [
            "Split the launch velocity into components: v0x = v0 cos(theta) and v0y = v0 sin(theta).",
            "Use the horizontal component for x-motion and the vertical component for y-motion. Keep gravity only in the vertical equation.",
            "Choose the equation that contains your unknown, then substitute only the known quantities first.",
        ]
    if concept_tag == "projectile_peak":
        return [
            "At the highest point, set the vertical velocity to zero: v_y = 0. That is the key condition for peak height.",
            "Use a vertical-motion equation such as v_y^2 = v0y^2 - 2g Delta y to relate the peak height to the initial vertical velocity.",
            "Solve for the height change first. If the projectile starts above the ground, add the starting height afterward.",
        ]
    if concept_tag == "range_formula":
        return [
            "Find the flight time from vertical motion first: y(t) = h0 + v0y t - (1/2)gt^2.",
            "Once you have the flight time, use horizontal motion: range = v0x t.",
            "Check that your time is positive and that units stay in seconds and meters.",
        ]
    if concept_tag == "newton_second_law":
        return [
            "Choose the object as your system and list every external force acting on it.",
            "Pick axes and write sum F = ma separately for each direction you need.",
            "Substitute known values only after the force equation is set up.",
        ]
    if concept_tag == "incline_components":
        return [
            "Choose axes parallel and perpendicular to the ramp. In chat, tell me which direction you chose as positive along the ramp.",
            "Break weight into components: mg sin(theta) along the ramp and mg cos(theta) perpendicular to the ramp.",
            "Write the net-force equation along the ramp after you decide whether friction is present.",
        ]
    if agent_id == "forces_agent":
        return [
            "List the forces acting on the object, like weight, normal force, tension, friction, or applied force.",
            "Choose axes and type the net-force equation in each direction.",
            "Substitute values after the equations are set up.",
        ]
    if agent_id == "kinematics_agent":
        return [
            "List the known quantities, the unknown quantity, and the positive direction.",
            "Choose the kinematics equation that includes the unknown and avoids extra unknowns.",
            "Substitute known values with units before solving.",
        ]
    return [
        "Identify the principle or equation that matches the concept check.",
        "List the known values and the unknown before substituting numbers.",
        "Try one algebra step, then send it back and I will check it.",
    ]


def _concept_setup_checklist(agent_id: str, concept_tag: str) -> list[str]:
    if concept_tag == "incline_components":
        return [
            "Your parallel axis should run along the ramp, and your perpendicular axis should point into or away from the ramp.",
            "Weight must point straight down, not perpendicular to the ramp.",
            "Normal force should be perpendicular to the surface, and friction should oppose the motion or possible motion.",
        ]
    if concept_tag == "hookes_law":
        return [
            "Your setup should identify k as the spring constant and x as the stretch or compression.",
            "The magnitude relation should multiply k and |x|, not divide them.",
            "The direction should be restoring, opposite the stretch or compression.",
        ]
    if concept_tag in {"projectile_components", "projectile_peak", "range_formula"}:
        return [
            "Your setup should split the initial velocity into horizontal and vertical components.",
            "Gravity belongs in the vertical equation only when air resistance is ignored.",
            "For maximum height, the vertical velocity at the top should be zero.",
        ]
    if agent_id == "forces_agent":
        return [
            "List only the external forces acting on the object you chose.",
            "Make sure each force has a direction.",
            "Write net-force equations after choosing axes.",
        ]
    if agent_id == "kinematics_agent":
        return [
            "List the known values and the unknown.",
            "State which direction is positive.",
            "Choose an equation that includes the unknown and avoids unnecessary extra unknowns.",
        ]
    return [
        "State the principle or equation you are using.",
        "List the known values and unknown.",
        "Show one setup step before calculating.",
    ]


def _coerce_sketch_point(raw_point: Any) -> Optional[Dict[str, float]]:
    if not isinstance(raw_point, dict):
        return None
    try:
        x = float(raw_point.get("x"))
        y = float(raw_point.get("y"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return {"x": x, "y": y}


def _coerce_sketch_strokes(raw_drawing: Any) -> list[Dict[str, Any]]:
    if not isinstance(raw_drawing, dict):
        return []
    raw_strokes = raw_drawing.get("strokes")
    if not isinstance(raw_strokes, list):
        return []

    strokes: list[Dict[str, Any]] = []
    for raw_stroke in raw_strokes[:120]:
        if not isinstance(raw_stroke, dict):
            continue
        raw_points = raw_stroke.get("points")
        if not isinstance(raw_points, list):
            continue
        points = [
            point
            for point in (_coerce_sketch_point(raw_point) for raw_point in raw_points[:700])
            if point is not None
        ]
        if points:
            strokes.append(
                {
                    "tool": str(raw_stroke.get("tool") or "pen"),
                    "color": str(raw_stroke.get("color") or "#111827"),
                    "points": points,
                }
            )
    return strokes


def _distance_between_points(a: Dict[str, float], b: Dict[str, float]) -> float:
    return math.hypot(float(b["x"]) - float(a["x"]), float(b["y"]) - float(a["y"]))


def _stroke_metrics(stroke: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    points = stroke.get("points")
    if not isinstance(points, list) or len(points) < 2:
        return None

    length = 0.0
    for i in range(1, len(points)):
        length += _distance_between_points(points[i - 1], points[i])
    first = points[0]
    last = points[-1]
    direct = _distance_between_points(first, last)
    xs = [float(point["x"]) for point in points]
    ys = [float(point["y"]) for point in points]
    x_extent = max(xs) - min(xs)
    y_extent = max(ys) - min(ys)
    dx = float(last["x"]) - float(first["x"])
    dy = float(last["y"]) - float(first["y"])

    long_enough = direct >= 45 or length >= 60
    is_horizontal = long_enough and x_extent >= 45 and y_extent <= max(14, x_extent * 0.28)
    is_vertical = long_enough and y_extent >= 45 and x_extent <= max(14, y_extent * 0.28)
    is_diagonal = long_enough and not is_horizontal and not is_vertical and x_extent >= 28 and y_extent >= 28
    curvature_ratio = length / max(direct, 1.0)
    is_curved = direct >= 35 and length >= 70 and curvature_ratio >= 1.18 and x_extent >= 35 and y_extent >= 24

    return {
        "tool": stroke.get("tool"),
        "length": round(length, 2),
        "direct_length": round(direct, 2),
        "angle_deg": round(math.degrees(math.atan2(-dy, dx)), 1),
        "min_x": min(xs),
        "max_x": max(xs),
        "min_y": min(ys),
        "max_y": max(ys),
        "mid_x": (min(xs) + max(xs)) / 2,
        "mid_y": (min(ys) + max(ys)) / 2,
        "is_horizontal": is_horizontal,
        "is_vertical": is_vertical,
        "is_diagonal": is_diagonal,
        "is_curved": is_curved,
        "is_force_like": long_enough and not is_curved,
    }


def _detect_axis_like_strokes(metrics: list[Dict[str, Any]]) -> bool:
    horizontal = [metric for metric in metrics if metric["is_horizontal"]]
    vertical = [metric for metric in metrics if metric["is_vertical"]]
    for h_metric in horizontal:
        for v_metric in vertical:
            crosses_horizontally = h_metric["min_x"] - 20 <= v_metric["mid_x"] <= h_metric["max_x"] + 20
            crosses_vertically = v_metric["min_y"] - 20 <= h_metric["mid_y"] <= v_metric["max_y"] + 20
            if crosses_horizontally and crosses_vertically:
                return True
    return bool(horizontal and vertical)


def _analyze_student_sketch(raw_drawing: Any, agent_id: str, concept_tag: str) -> Optional[Dict[str, Any]]:
    strokes = _coerce_sketch_strokes(raw_drawing)
    if not strokes:
        return None

    eraser_stroke_count = sum(1 for stroke in strokes if stroke.get("tool") == "eraser")
    pen_metrics = [
        metric
        for metric in (_stroke_metrics(stroke) for stroke in strokes if stroke.get("tool") != "eraser")
        if metric is not None
    ]
    if not pen_metrics:
        return {
            "status": "needs_more_detail",
            "stroke_count": 0,
            "eraser_stroke_count": eraser_stroke_count,
            "strengths": [],
            "suggestions": ["Add the main axes, vectors, or trajectory with the pen tool before attaching the sketch."],
            "limitations": ["Sketch analysis reads line geometry, not handwritten labels."],
        }

    horizontal_count = sum(1 for metric in pen_metrics if metric["is_horizontal"])
    vertical_count = sum(1 for metric in pen_metrics if metric["is_vertical"])
    diagonal_count = sum(1 for metric in pen_metrics if metric["is_diagonal"])
    curved_count = sum(1 for metric in pen_metrics if metric["is_curved"])
    force_like_count = sum(1 for metric in pen_metrics if metric["is_force_like"])
    axis_like = _detect_axis_like_strokes(pen_metrics)

    strengths: list[str] = []
    suggestions: list[str] = []

    if axis_like:
        strengths.append("I can see an x/y-axis structure in the sketch.")
    if diagonal_count:
        strengths.append("I can see at least one diagonal line, which may represent a ramp, launch direction, or angled vector.")
    if curved_count:
        strengths.append("I can see a curved path, which fits a projectile trajectory sketch.")

    if agent_id == "kinematics_agent":
        if concept_tag in {"projectile_components", "projectile_peak", "range_formula"}:
            if not axis_like:
                suggestions.append("Add clear horizontal and vertical axes so the projectile components are tied to directions.")
            if not diagonal_count and not curved_count:
                suggestions.append("Add either the launch-direction vector or a curved trajectory path.")
            suggestions.append("Type the velocity labels, angle, starting height, and which direction is positive.")
        else:
            if not axis_like and horizontal_count == 0:
                suggestions.append("Add a position or time axis to show the 1D motion direction.")
            suggestions.append("Type your known values, unknown, and the equation you chose.")
    elif agent_id == "forces_agent":
        if concept_tag == "incline_components" and diagonal_count:
            strengths.append("The diagonal line is a reasonable starting point for an inclined-plane sketch.")
        if force_like_count < 3:
            suggestions.append("For a free-body diagram, add separate arrows for weight, normal force, and friction/tension/applied force when present.")
        if concept_tag == "incline_components" and not axis_like:
            suggestions.append("Add axes parallel and perpendicular to the ramp.")
        suggestions.append("Type the force labels because the sketch analyzer cannot reliably read handwriting yet.")

    if eraser_stroke_count:
        suggestions.append("If you erased major parts, use Clear and redraw before attaching for the most accurate analysis.")
    if not suggestions:
        suggestions.append("Type the labels and one setup equation so I can check the physics, not just the geometry.")

    confidence = min(
        0.95,
        0.30
        + 0.08 * min(force_like_count, 5)
        + (0.18 if axis_like else 0.0)
        + (0.12 if diagonal_count else 0.0)
        + (0.12 if curved_count else 0.0),
    )

    return {
        "status": "analyzed",
        "stroke_count": len(pen_metrics),
        "eraser_stroke_count": eraser_stroke_count,
        "line_count": force_like_count,
        "axis_like": axis_like,
        "horizontal_count": horizontal_count,
        "vertical_count": vertical_count,
        "diagonal_count": diagonal_count,
        "curved_count": curved_count,
        "confidence": round(confidence, 2),
        "strengths": strengths,
        "suggestions": suggestions,
        "limitations": ["Sketch analysis reads stroke geometry, not handwritten labels."],
    }


def _format_sketch_analysis_feedback(sketch_analysis: Optional[Dict[str, Any]]) -> str:
    if not sketch_analysis:
        return ""

    lines = ["Sketch analysis:"]
    for strength in sketch_analysis.get("strengths") or []:
        lines.append(f"- {strength}")
    for suggestion in sketch_analysis.get("suggestions") or []:
        lines.append(f"- {suggestion}")
    for limitation in sketch_analysis.get("limitations") or []:
        lines.append(f"- {limitation}")
    return "\n".join(lines)


def _remediation_followup_intent(student_message: str) -> Dict[str, bool]:
    lower_message = (student_message or "").strip().lower()
    placeholder_only = lower_message == "attached a sketch of my work."
    message_words = {
        word.strip(".,!?;:\"'()[]{}")
        for word in lower_message.split()
        if word.strip(".,!?;:\"'()[]{}")
    }
    asks_for_next_step = any(phrase in lower_message for phrase in ("next", "step", "hint", "continue"))
    asks_for_explanation = any(
        phrase in lower_message
        for phrase in ("don't understand", "dont understand", "do not understand", "not understand")
    ) or bool({"why", "explain", "no"} & message_words)
    asks_for_full_solution = any(
        phrase in lower_message
        for phrase in (
            "full solution",
            "show solution",
            "show me the solution",
            "give me the solution",
            "give solution",
            "i am lost",
            "i'm lost",
            "im lost",
            "completely lost",
            "really lost",
            "stuck",
        )
    )
    understands = any(phrase in lower_message for phrase in ("yes", "understand", "got it", "makes sense"))
    submitted_work = (
        bool(lower_message)
        and not placeholder_only
        and not asks_for_next_step
        and not asks_for_explanation
        and not asks_for_full_solution
        and not understands
    )
    return {
        "asks_for_next_step": asks_for_next_step,
        "asks_for_explanation": asks_for_explanation,
        "asks_for_full_solution": asks_for_full_solution,
        "understands": understands,
        "submitted_work": submitted_work,
    }


def _normalize_math_submission(text: str) -> str:
    normalized = (text or "").lower()
    normalized = re.sub(r"\\frac\s*\{\s*1\s*\}\s*\{\s*2\s*\}", "1/2", normalized)
    replacements = {
        "\\delta": "delta",
        "Δ": "delta",
        "δ": "delta",
        "\\theta": "theta",
        "θ": "theta",
        "\\sum": "sum",
        "Σ": "sum",
        "∑": "sum",
        "\\vec": "vec",
        "\\overrightarrow": "vec",
        "\\cdot": "",
        "\\times": "",
        "\\left": "",
        "\\right": "",
        "\\ ": "",
        "½": "1/2",
        "initial": "i",
        "final": "f",
    }
    for old, new in replacements.items():
        normalized = normalized.replace(old.lower(), new)
    normalized = normalized.replace("{", "").replace("}", "")
    normalized = normalized.replace("_", "")
    normalized = normalized.replace(" ", "")
    normalized = normalized.replace("*", "")
    normalized = normalized.replace("\\", "")
    return normalized


def _numeric_literals(text: str) -> list[str]:
    cleaned = re.sub(r"\\frac\s*\{\s*1\s*\}\s*\{\s*2\s*\}", "", text or "")
    cleaned = re.sub(r"\^\s*\{?\s*[-+]?\d+(?:\.\d+)?\s*\}?", "", cleaned)
    return re.findall(r"(?<![A-Za-z_\\^/{])[-+]?\d+(?:\.\d+)?", cleaned)


def _safe_float(value: Optional[str]) -> Optional[float]:
    try:
        parsed = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    if parsed is None or not math.isfinite(parsed):
        return None
    return parsed


def _format_number(value: float) -> str:
    return f"{value:g}"


def _message_contains_number_close_to(text: str, expected: float) -> bool:
    tolerance = max(0.05, abs(expected) * 0.02)
    for literal in _numeric_literals(text):
        value = _safe_float(literal)
        if value is not None and abs(value - expected) <= tolerance:
            return True
    return False


def _first_number_match(text: str, patterns: list[str]) -> Optional[str]:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _extract_kinematics_knowns(problem: str) -> Dict[str, str]:
    lower_problem = (problem or "").lower()
    knowns: Dict[str, str] = {}

    if re.search(r"\b(?:from rest|starts? from rest|initially at rest)\b", lower_problem):
        knowns["v0"] = "0"
    else:
        initial_velocity = _first_number_match(
            problem,
            [
                r"(?:initial\s+(?:velocity|speed)|v[_\s]*0|v0|vi)\s*(?:=|is|of|:)?\s*(-?\d+(?:\.\d+)?)\s*(?:m\s*/\s*s|m/s|mps|meters?\s+per\s+second)",
                r"(?:starts?|begins)\s+(?:at|with)\s*(-?\d+(?:\.\d+)?)\s*(?:m\s*/\s*s|m/s|mps|meters?\s+per\s+second)",
            ],
        )
        if initial_velocity is not None:
            knowns["v0"] = initial_velocity

    acceleration = _first_number_match(
        problem,
        [
            r"\baccelerat(?:es?|ed|ing|ion)?(?:\s+uniformly)?\s*(?:at|=|of|:)?\s*(-?\d+(?:\.\d+)?)\s*(?:m\s*/\s*s(?:\^?2|²)|m/s2|mps2)",
            r"\ba\s*(?:=|is|:)\s*(-?\d+(?:\.\d+)?)\s*(?:m\s*/\s*s(?:\^?2|²)|m/s2|mps2)",
        ],
    )
    if acceleration is not None:
        knowns["a"] = acceleration

    time_value = _first_number_match(
        problem,
        [
            r"(?:for|time(?:\s+of)?|in|t\s*=)\s*(-?\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\b",
            r"\b(-?\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\b",
        ],
    )
    if time_value is not None:
        knowns["t"] = time_value

    displacement = _first_number_match(
        problem,
        [
            r"(?:displacement|distance|delta\s*x|height)\s*(?:=|is|of|:)?\s*(-?\d+(?:\.\d+)?)\s*m\b",
            r"\b(-?\d+(?:\.\d+)?)\s*m\s+(?:cliff|height|balcony|above|high)\b",
        ],
    )
    if displacement is not None:
        knowns["dx"] = displacement

    return knowns


def _kinematics_substitution_example(problem: str, family: str) -> Optional[str]:
    knowns = _extract_kinematics_knowns(problem)
    if family == "final_velocity_time" and {"v0", "a", "t"} <= knowns.keys():
        return f"v_f = {knowns['v0']} + ({knowns['a']})({knowns['t']})"
    if family == "displacement_time" and {"v0", "a", "t"} <= knowns.keys():
        return f"Delta x = ({knowns['v0']})({knowns['t']}) + (1/2)({knowns['a']})({knowns['t']})^2"
    if family == "final_velocity_no_time" and {"v0", "a", "dx"} <= knowns.keys():
        return f"v_f^2 = ({knowns['v0']})^2 + 2({knowns['a']})({knowns['dx']})"
    return None


def _kinematics_substitution_instruction(problem: str, family: str) -> str:
    example = _kinematics_substitution_example(problem, family)
    if family == "final_velocity_time":
        instruction = "Replace the symbols with the values from the problem in v_f = v_0 + at."
    elif family == "displacement_time":
        instruction = "Replace the symbols with the values from the problem in Delta x = v_0 t + (1/2) a t^2."
    elif family == "final_velocity_no_time":
        instruction = "Replace the symbols with the values from the problem in v_f^2 = v_0^2 + 2a Delta x."
    else:
        instruction = "Replace the symbols with the numerical values from the problem."
    if example:
        return f"{instruction} For this problem, that starts as: {example}."
    return instruction


def _expected_kinematics_result(problem: str, family: str) -> Optional[Dict[str, Any]]:
    knowns = _extract_kinematics_knowns(problem)
    v0 = _safe_float(knowns.get("v0"))
    acceleration = _safe_float(knowns.get("a"))
    time_value = _safe_float(knowns.get("t"))
    displacement = _safe_float(knowns.get("dx"))

    if family == "final_velocity_time" and v0 is not None and acceleration is not None and time_value is not None:
        return {
            "label": "v_f",
            "value": v0 + acceleration * time_value,
            "units": "m/s",
        }
    if family == "displacement_time" and v0 is not None and acceleration is not None and time_value is not None:
        return {
            "label": "Delta x",
            "value": v0 * time_value + 0.5 * acceleration * time_value**2,
            "units": "m",
        }
    if family == "final_velocity_no_time" and v0 is not None and acceleration is not None and displacement is not None:
        radicand = v0**2 + 2 * acceleration * displacement
        if radicand >= 0:
            return {
                "label": "v_f",
                "value": math.sqrt(radicand),
                "units": "m/s",
            }
    return None


def _format_kinematics_result(result: Dict[str, Any]) -> str:
    return f"{result['label']} = {_format_number(float(result['value']))} {result['units']}"


def _mentions_kinematics_var(normalized: str, lower_message: str, variable: str) -> bool:
    if variable == "vf":
        return "vf" in normalized or "v_f" in lower_message or "final velocity" in lower_message or "final speed" in lower_message
    if variable == "v0":
        return (
            "v0" in normalized
            or "v_0" in lower_message
            or "initial velocity" in lower_message
            or "initial speed" in lower_message
            or "from rest" in lower_message
            or "at rest" in lower_message
        )
    if variable == "a":
        return bool(re.search(r"(?<![A-Za-z])a(?![A-Za-z])", lower_message)) or "accel" in lower_message
    if variable == "t":
        return bool(re.search(r"(?<![A-Za-z])t(?![A-Za-z])", lower_message)) or "time" in lower_message
    if variable == "dx":
        return (
            "deltax" in normalized
            or "dx" in normalized
            or "delta x" in lower_message
            or "displacement" in lower_message
            or "distance" in lower_message
        )
    if variable == "theta":
        return "theta" in normalized or "angle" in lower_message or "°" in lower_message
    return False


def _mentions_all_kinematics_vars(normalized: str, lower_message: str, variables: tuple[str, ...]) -> bool:
    return all(_mentions_kinematics_var(normalized, lower_message, variable) for variable in variables)


def _mentions_any_kinematics_vars(normalized: str, lower_message: str, variables: tuple[str, ...]) -> bool:
    return any(_mentions_kinematics_var(normalized, lower_message, variable) for variable in variables)


def _problem_kinematics_family(problem: str) -> str:
    lower = (problem or "").lower()
    asks_final_velocity = bool(re.search(r"\b(final velocity|final speed|v_f|vf)\b", lower))
    asks_displacement = bool(re.search(r"\b(displacement|distance|position|how far)\b", lower))
    asks_time = bool(re.search(r"\b(time|how long|flight time)\b", lower))
    has_time = bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:s|sec|secs|second|seconds)\b", lower))
    has_acceleration = bool(re.search(r"\baccelerat(?:e|es|ed|ing|ion)\b|\bm/s\^?2\b", lower))
    has_distance = bool(re.search(r"\b\d+(?:\.\d+)?\s*m\b", lower)) and not bool(re.search(r"\b\d+(?:\.\d+)?\s*m/s\b", lower))

    if asks_final_velocity and has_acceleration and has_time:
        return "final_velocity_time"
    if asks_displacement and has_acceleration and has_time:
        return "displacement_time"
    if asks_final_velocity and has_acceleration and has_distance and not has_time:
        return "final_velocity_no_time"
    if asks_time:
        return "solve_for_time"
    return "general"


def _has_final_velocity_time_equation(normalized: str) -> bool:
    has_final_velocity = any(token in normalized for token in ("vf", "vfinal", "v="))
    has_initial_velocity = any(token in normalized for token in ("v0", "vi", "u"))
    has_accel_time = "at" in normalized or "a*t" in normalized
    return "=" in normalized and has_final_velocity and has_initial_velocity and has_accel_time


def _has_displacement_time_equation(normalized: str) -> bool:
    has_displacement = any(token in normalized for token in ("deltax", "dx", "x=", "s="))
    has_initial_term = any(token in normalized for token in ("v0t", "vit", "ut"))
    has_half_accel_term = any(token in normalized for token in ("1/2at^2", "0.5at^2", ".5at^2", "at^2/2"))
    return "=" in normalized and has_displacement and (has_initial_term or has_half_accel_term) and has_half_accel_term


def _has_no_time_velocity_equation(normalized: str) -> bool:
    has_final_squared = "vf^2" in normalized or "v^2" in normalized
    has_initial_squared = any(token in normalized for token in ("v0^2", "vi^2", "u^2"))
    has_accel_displacement = (
        "2a" in normalized
        and any(token in normalized for token in ("deltax", "dx", "x", "s"))
    )
    return "=" in normalized and has_final_squared and has_initial_squared and has_accel_displacement


def _has_projectile_component_setup(normalized: str, lower_message: str) -> bool:
    has_trig = "cos" in normalized and "sin" in normalized
    has_horizontal = any(token in normalized for token in ("v0x", "vx")) or "horizontal" in lower_message
    has_vertical = any(token in normalized for token in ("v0y", "vy")) or "vertical" in lower_message
    return has_trig and has_horizontal and has_vertical


def _has_projectile_peak_condition(normalized: str, lower_message: str) -> bool:
    return (
        "vy=0" in normalized
        or "vfy=0" in normalized
        or "verticalvelocityiszero" in normalized
        or ("vertical velocity" in lower_message and "zero" in lower_message)
    )


def _has_projectile_range_setup(normalized: str, lower_message: str) -> bool:
    uses_vertical_time = "flight time" in lower_message or ("vertical" in lower_message and "time" in lower_message)
    uses_horizontal_range = "range" in lower_message or "horizontal" in lower_message or "x=v0xt" in normalized
    return uses_vertical_time and uses_horizontal_range


def _next_step_after_correct_submission(agent_id: str, concept_tag: str, problem: str, step_index: int = 0) -> str:
    if agent_id == "kinematics_agent":
        family = _problem_kinematics_family(problem)
        if step_index >= 1:
            if concept_tag == "projectile_components":
                return "Now calculate the numerical x- and y-components with units, then send those two values only."
            if concept_tag == "projectile_peak":
                return "Now solve the vertical-motion equation for the height change, keeping the sign of g visible."
            if concept_tag == "range_formula":
                return "Now solve the vertical-motion equation for the positive flight time before using horizontal range."
            if family == "final_velocity_time":
                return "Now calculate a t, add it to v_0, and write v_f with units."
            if family == "displacement_time":
                return "Now calculate each term, v_0 t and (1/2) a t^2, then add them with units."
            if family == "final_velocity_no_time":
                return "Now calculate the right side first, then take the square root and keep the physical sign."
            return "Now calculate the next algebra step, keeping units and signs visible."
        if concept_tag == "projectile_components":
            return "Substitute the launch speed and angle into v0x = v0 cos(theta) and v0y = v0 sin(theta), keeping units."
        if concept_tag == "projectile_peak":
            return "Use a vertical-motion equation with v_y = 0 at the peak, and solve only for the height change first."
        if concept_tag == "range_formula":
            return "Set up the vertical-motion equation to find flight time first, then use horizontal motion for range."
        if family == "final_velocity_time":
            return "Substitute v_0, a, and t from the problem into v_f = v_0 + at, keeping units."
        if family == "displacement_time":
            return "Substitute v_0, a, and t into Delta x = v_0 t + (1/2) a t^2, keeping units."
        if family == "final_velocity_no_time":
            return "Substitute v_0, a, and Delta x into v_f^2 = v_0^2 + 2a Delta x, then pause before taking the square root."
        return "List the known values under your chosen equation, including units and signs."

    if agent_id == "forces_agent":
        if concept_tag == "hookes_law":
            return "Substitute k and x into |F_s| = k|x|, then decide the restoring direction separately."
        if concept_tag == "incline_components":
            return "Write the net-force equation along the ramp using your chosen positive direction."
        return "Write sum F = ma for the chosen object and direction, then substitute the known forces."

    return "Substitute the known values into your setup, keeping units and signs visible."


def _check_kinematics_substitution_step(
    concept_tag: str,
    problem: str,
    student_message: str,
    normalized: str,
    lower_message: str,
) -> Optional[Dict[str, Any]]:
    family = _problem_kinematics_family(problem)
    numbers = _numeric_literals(student_message)
    has_equation = "=" in normalized
    has_numeric_substitution = len(numbers) >= 2

    if concept_tag == "projectile_components":
        has_symbols_only = _mentions_all_kinematics_vars(normalized, lower_message, ("v0", "theta"))
        if has_numeric_substitution and ("sin" in normalized or "cos" in normalized or has_equation):
            return {
                "status": "correct",
                "feedback": "Good: you started substituting numerical values into the component equations.",
                "next_step": _next_step_after_correct_submission("kinematics_agent", concept_tag, problem, step_index=1),
            }
        if _has_projectile_component_setup(normalized, lower_message) or has_symbols_only:
            return {
                "status": "partial",
                "feedback": "You identified the right component symbols, but the substitution step needs the actual launch speed and angle from the problem.",
                "next_step": "Replace v_0 and theta with the numerical values from the prompt in both component equations.",
            }
        return None

    family_variables = {
        "final_velocity_time": ("v0", "a", "t"),
        "displacement_time": ("v0", "a", "t"),
        "final_velocity_no_time": ("v0", "a", "dx"),
    }
    expected_variables = family_variables.get(family)
    if not expected_variables:
        return None

    substitution_instruction = _kinematics_substitution_instruction(problem, family)
    has_expected_symbols = _mentions_all_kinematics_vars(normalized, lower_message, expected_variables)
    has_any_expected_symbol = _mentions_any_kinematics_vars(normalized, lower_message, expected_variables)

    if family == "final_velocity_time":
        if _has_displacement_time_equation(normalized):
            return {
                "status": "incorrect",
                "feedback": "That is the displacement equation, but this step is substituting into the final-velocity equation.",
                "next_step": substitution_instruction,
                "expected": "v_f = v_0 + a t",
            }
        if _has_no_time_velocity_equation(normalized):
            return {
                "status": "partial",
                "feedback": "The squared-velocity equation is not the most direct path because time is given.",
                "next_step": substitution_instruction,
                "expected": "v_f = v_0 + a t",
            }
        if has_numeric_substitution and (has_equation or has_any_expected_symbol):
            return {
                "status": "correct",
                "feedback": "Good: you substituted numerical values for the known quantities in the final-velocity setup.",
                "next_step": _next_step_after_correct_submission("kinematics_agent", concept_tag, problem, step_index=1),
                "expected": "v_f = v_0 + a t",
            }
        if _has_final_velocity_time_equation(normalized):
            return {
                "status": "correct",
                "feedback": "That equation is correct. The next move is to substitute the actual values from the problem.",
                "next_step": substitution_instruction,
                "expected": "v_f = v_0 + a t",
                "advance_step": False,
            }
        if has_expected_symbols:
            return {
                "status": "partial",
                "feedback": "You identified the right symbols, but substitution means replacing v_0, a, and t with the numbers from the problem.",
                "next_step": substitution_instruction,
                "expected": "v_f = v_0 + a t",
            }
        if numbers:
            return {
                "status": "partial",
                "feedback": "I see some numbers, but I need each one tied to v_0, a, and t in the final-velocity equation.",
                "next_step": substitution_instruction,
                "expected": "v_f = v_0 + a t",
            }
        return None

    if family == "displacement_time":
        if _has_final_velocity_time_equation(normalized):
            return {
                "status": "incorrect",
                "feedback": "That equation finds final velocity, but this step is substituting into the displacement equation.",
                "next_step": substitution_instruction,
                "expected": "Delta x = v_0 t + (1/2) a t^2",
            }
        if has_numeric_substitution and (has_equation or has_any_expected_symbol):
            return {
                "status": "correct",
                "feedback": "Good: you substituted numerical values into the displacement equation.",
                "next_step": _next_step_after_correct_submission("kinematics_agent", concept_tag, problem, step_index=1),
                "expected": "Delta x = v_0 t + (1/2) a t^2",
            }
        if _has_displacement_time_equation(normalized):
            return {
                "status": "correct",
                "feedback": "That equation is correct. The next move is to substitute the actual values from the problem.",
                "next_step": substitution_instruction,
                "expected": "Delta x = v_0 t + (1/2) a t^2",
                "advance_step": False,
            }
        if has_expected_symbols:
            return {
                "status": "partial",
                "feedback": "You identified the right symbols, but this step needs the numerical values for v_0, a, and t.",
                "next_step": substitution_instruction,
                "expected": "Delta x = v_0 t + (1/2) a t^2",
            }
        return None

    if family == "final_velocity_no_time":
        if has_numeric_substitution and (has_equation or has_any_expected_symbol):
            return {
                "status": "correct",
                "feedback": "Good: you substituted numerical values into the no-time velocity equation.",
                "next_step": _next_step_after_correct_submission("kinematics_agent", concept_tag, problem, step_index=1),
                "expected": "v_f^2 = v_0^2 + 2 a Delta x",
            }
        if _has_no_time_velocity_equation(normalized):
            return {
                "status": "correct",
                "feedback": "That equation is correct. The next move is to substitute the actual values from the problem.",
                "next_step": substitution_instruction,
                "expected": "v_f^2 = v_0^2 + 2 a Delta x",
                "advance_step": False,
            }
        if has_expected_symbols:
            return {
                "status": "partial",
                "feedback": "You identified the right symbols, but this step needs the numerical values for v_0, a, and Delta x.",
                "next_step": substitution_instruction,
                "expected": "v_f^2 = v_0^2 + 2 a Delta x",
            }
        return None

    return None


def _check_kinematics_calculation_step(
    concept_tag: str,
    problem: str,
    student_message: str,
    normalized: str,
    lower_message: str,
) -> Optional[Dict[str, Any]]:
    family = _problem_kinematics_family(problem)
    expected_result = _expected_kinematics_result(problem, family)
    numbers = _numeric_literals(student_message)

    if concept_tag == "projectile_components":
        if numbers and ("vx" in normalized or "vy" in normalized or "component" in lower_message):
            return {
                "status": "correct",
                "feedback": "Good: you are calculating the projectile components and labeling them by direction.",
                "next_step": "Use vertical motion to find the time in the air, then use horizontal motion for range.",
            }
        if numbers:
            return {
                "status": "partial",
                "feedback": "I see numerical work, but I need the component labels so I know which value is horizontal and which is vertical.",
                "next_step": "Write v_0x and v_0y with their numerical values and units.",
            }
        return None

    if not expected_result:
        return None

    expected_value = float(expected_result["value"])
    expected_text = _format_kinematics_result(expected_result)
    has_expected_number = _message_contains_number_close_to(student_message, expected_value)

    if family == "final_velocity_time":
        mentions_final_velocity = _mentions_kinematics_var(normalized, lower_message, "vf")
        if has_expected_number and (mentions_final_velocity or "m/s" in lower_message or "meter per second" in lower_message):
            return {
                "status": "complete",
                "feedback": f"Correct: the final velocity is {expected_text}.",
                "next_step": "This problem is complete. You can ask for a graph, ask a follow-up question, or start a new problem.",
                "expected": expected_text,
            }
        if has_expected_number:
            return {
                "status": "partial",
                "feedback": "Your calculation has the right numerical value, but it needs to be labeled as the final velocity with units.",
                "next_step": f"Write the final answer as {expected_text}.",
                "expected": expected_text,
            }
        if numbers:
            return {
                "status": "incorrect",
                "feedback": f"The arithmetic does not match the known values. The final value should be {expected_text}.",
                "next_step": "Recalculate a t first, then add it to v_0 and include units.",
                "expected": expected_text,
            }
        return {
            "status": "partial",
            "feedback": "I need to see the numerical calculation for the final velocity.",
            "next_step": "Calculate a t, add it to v_0, and write v_f with units.",
            "expected": expected_text,
        }

    if family == "displacement_time":
        if has_expected_number and (
            _mentions_kinematics_var(normalized, lower_message, "dx")
            or "meter" in lower_message
            or " m" in lower_message
        ):
            return {
                "status": "complete",
                "feedback": f"Correct: the displacement is {expected_text}.",
                "next_step": "This problem is complete. You can ask for a graph, ask a follow-up question, or start a new problem.",
                "expected": expected_text,
            }
        if has_expected_number:
            return {
                "status": "partial",
                "feedback": "Your calculation has the right numerical value, but it needs a displacement label and units.",
                "next_step": f"Write the final answer as {expected_text}.",
                "expected": expected_text,
            }
        if numbers:
            return {
                "status": "incorrect",
                "feedback": f"The arithmetic does not match the known values. The displacement should be {expected_text}.",
                "next_step": "Recalculate v_0 t and (1/2) a t^2 separately, then add them.",
                "expected": expected_text,
            }
        return None

    if family == "final_velocity_no_time":
        if has_expected_number and (
            _mentions_kinematics_var(normalized, lower_message, "vf")
            or "m/s" in lower_message
            or "meter per second" in lower_message
        ):
            return {
                "status": "complete",
                "feedback": f"Correct: the final velocity magnitude is {expected_text}.",
                "next_step": "This problem is complete. You can ask for a graph, ask a follow-up question, or start a new problem.",
                "expected": expected_text,
            }
        if has_expected_number:
            return {
                "status": "partial",
                "feedback": "Your numerical value is right, but label it as final velocity and include units.",
                "next_step": f"Write the final answer as {expected_text}.",
                "expected": expected_text,
            }
        if numbers:
            return {
                "status": "incorrect",
                "feedback": f"The arithmetic or square root does not match the known values. The final velocity magnitude should be {expected_text}.",
                "next_step": "Recalculate the right side of the squared-velocity equation before taking the square root.",
                "expected": expected_text,
            }
        return None

    return None


def _check_student_submission(
    agent_id: str,
    concept_tag: str,
    problem: str,
    student_message: str,
    sketch_analysis: Optional[Dict[str, Any]] = None,
    step_index: int = 0,
) -> Dict[str, Any]:
    lower_message = (student_message or "").strip().lower()
    normalized = _normalize_math_submission(student_message)
    has_meaningful_text = bool(lower_message) and lower_message != "attached a sketch of my work."

    if not has_meaningful_text:
        return {
            "status": "needs_text",
            "feedback": "I can check the sketch geometry, but I still need the equation, labels, or force names typed in chat.",
            "next_step": "Type the equation or labels from your sketch so I can check them directly.",
        }

    if agent_id == "kinematics_agent":
        calculation_check = _check_kinematics_calculation_step(
            concept_tag=concept_tag,
            problem=problem,
            student_message=student_message,
            normalized=normalized,
            lower_message=lower_message,
        )
        if calculation_check and calculation_check.get("status") == "complete":
            return calculation_check
        if step_index >= 2 and calculation_check:
            return calculation_check

        if step_index == 1:
            substitution_check = _check_kinematics_substitution_step(
                concept_tag=concept_tag,
                problem=problem,
                student_message=student_message,
                normalized=normalized,
                lower_message=lower_message,
            )
            if substitution_check:
                return substitution_check

        if concept_tag == "projectile_components":
            if _has_projectile_component_setup(normalized, lower_message):
                return {
                    "status": "correct",
                    "feedback": "Your component setup is right: use cosine for the horizontal component and sine for the vertical component.",
                    "next_step": _next_step_after_correct_submission(agent_id, concept_tag, problem, step_index=step_index),
                }
            if "sin" in normalized or "cos" in normalized:
                return {
                    "status": "partial",
                    "feedback": "You are using trig, which is good, but I need to see both v0x = v0 cos(theta) and v0y = v0 sin(theta).",
                    "next_step": "Write both component equations explicitly before solving.",
                }
            return {
                "status": "incorrect",
                "feedback": "For an angled launch, the full launch speed is not all horizontal or all vertical.",
                "next_step": "Start by splitting the launch velocity into x and y components.",
            }

        if concept_tag == "projectile_peak":
            if _has_projectile_peak_condition(normalized, lower_message):
                return {
                    "status": "correct",
                    "feedback": "Correct: the vertical velocity is zero at the highest point.",
                    "next_step": _next_step_after_correct_submission(agent_id, concept_tag, problem, step_index=step_index),
                }
            return {
                "status": "incorrect",
                "feedback": "At the peak, acceleration is still downward; the condition that becomes zero is vertical velocity.",
                "next_step": "Write v_y = 0 at the highest point before choosing the vertical-motion equation.",
            }

        if concept_tag == "range_formula":
            if _has_projectile_range_setup(normalized, lower_message):
                return {
                    "status": "correct",
                    "feedback": "Correct: use vertical motion to find flight time, then horizontal motion to find range.",
                    "next_step": _next_step_after_correct_submission(agent_id, concept_tag, problem, step_index=step_index),
                }
            return {
                "status": "partial",
                "feedback": "For range, the key is linking vertical flight time to horizontal motion.",
                "next_step": "First set up the vertical-motion equation for time in the air.",
            }

        family = _problem_kinematics_family(problem)
        if family == "final_velocity_time":
            if _has_final_velocity_time_equation(normalized):
                return {
                    "status": "correct",
                    "feedback": "That is the right equation for final velocity when acceleration and time are given.",
                    "next_step": _next_step_after_correct_submission(agent_id, concept_tag, problem, step_index=step_index),
                    "expected": "v_f = v_0 + a t",
                }
            if _has_displacement_time_equation(normalized):
                return {
                    "status": "incorrect",
                    "feedback": "That displacement equation is useful for Delta x, but this problem asks for final velocity.",
                    "next_step": "Use v_f = v_0 + at for this question.",
                    "expected": "v_f = v_0 + a t",
                }
            if _has_no_time_velocity_equation(normalized):
                return {
                    "status": "partial",
                    "feedback": "The squared-velocity equation can work when displacement is known, but this problem gives time directly.",
                    "next_step": "Use v_f = v_0 + at because the problem gives acceleration and time.",
                    "expected": "v_f = v_0 + a t",
                }
            return {
                "status": "incorrect",
                "feedback": "I do not see the final-velocity equation that uses acceleration and time.",
                "next_step": "Write v_f = v_0 + at.",
                "expected": "v_f = v_0 + a t",
            }

        if family == "displacement_time":
            if _has_displacement_time_equation(normalized):
                return {
                    "status": "correct",
                    "feedback": "That is the right displacement equation for constant acceleration with time given.",
                    "next_step": _next_step_after_correct_submission(agent_id, concept_tag, problem, step_index=step_index),
                    "expected": "Delta x = v_0 t + (1/2) a t^2",
                }
            return {
                "status": "incorrect",
                "feedback": "This problem asks for displacement, so the equation should include Delta x and t^2.",
                "next_step": "Use Delta x = v_0 t + (1/2) a t^2.",
                "expected": "Delta x = v_0 t + (1/2) a t^2",
            }

        if family == "final_velocity_no_time":
            if _has_no_time_velocity_equation(normalized):
                return {
                    "status": "correct",
                    "feedback": "That is the right no-time equation connecting velocity, acceleration, and displacement.",
                    "next_step": _next_step_after_correct_submission(agent_id, concept_tag, problem, step_index=step_index),
                    "expected": "v_f^2 = v_0^2 + 2 a Delta x",
                }
            return {
                "status": "incorrect",
                "feedback": "Time is not the useful known here, so use the squared-velocity relation.",
                "next_step": "Write v_f^2 = v_0^2 + 2a Delta x.",
                "expected": "v_f^2 = v_0^2 + 2 a Delta x",
            }

        if any(
            checker(normalized)
            for checker in (_has_final_velocity_time_equation, _has_displacement_time_equation, _has_no_time_velocity_equation)
        ):
            return {
                "status": "partial",
                "feedback": "You wrote a valid kinematics equation. I still need to match it to the exact unknown and known values in this problem.",
                "next_step": "Tell me the unknown and the known values you are using with that equation.",
            }
        return {
            "status": "partial",
            "feedback": "I can see you are setting up the problem, but I do not see a complete kinematics equation yet.",
            "next_step": "Write the equation you plan to use with the unknown on the left if possible.",
        }

    if agent_id == "forces_agent":
        if concept_tag == "hookes_law":
            if "x/k" in normalized or "k/x" in normalized:
                return {
                    "status": "incorrect",
                    "feedback": "That divides instead of multiplying; Hooke's law is linear in displacement.",
                    "next_step": "Write |F_s| = k|x|.",
                    "expected": "|F_s| = k|x|",
                }
            if "kx" in normalized and "=" in normalized:
                return {
                    "status": "correct",
                    "feedback": "Correct: spring force magnitude is k times the stretch or compression.",
                    "next_step": _next_step_after_correct_submission(agent_id, concept_tag, problem),
                    "expected": "|F_s| = k|x|",
                }

        if concept_tag == "incline_components":
            has_axes = any(word in lower_message for word in ("parallel", "perpendicular", "along the ramp", "normal"))
            has_components = "mgsin" in normalized and "mgcos" in normalized
            if has_axes and has_components:
                return {
                    "status": "correct",
                    "feedback": "Correct: incline-aligned axes and mg sin(theta)/mg cos(theta) are the right setup.",
                    "next_step": _next_step_after_correct_submission(agent_id, concept_tag, problem),
                }
            if has_axes or has_components or sketch_analysis:
                return {
                    "status": "partial",
                    "feedback": "The setup is partly there, but I need both ramp-aligned axes and the weight components mg sin(theta), mg cos(theta).",
                    "next_step": "Add the missing axis/component information before writing the net-force equation.",
                }

        if "sumf=ma" in normalized or "fnet=ma" in normalized or ("net force" in lower_message and "ma" in normalized):
            return {
                "status": "correct",
                "feedback": "Correct: Newton's second law must use the net external force on the object.",
                "next_step": _next_step_after_correct_submission(agent_id, concept_tag, problem),
                "expected": "sum F = m a",
            }

        return {
            "status": "partial",
            "feedback": "I do not yet see a complete force setup with the object, forces, and net-force equation.",
            "next_step": "List each external force on the object, then write sum F = ma for the chosen direction.",
        }

    return {
        "status": "partial",
        "feedback": "I can see your setup, but I need one explicit equation or principle to check it.",
        "next_step": "Write the equation or principle you want to use.",
    }


def _format_submission_check_response(
    submission_check: Dict[str, Any],
    sketch_analysis: Optional[Dict[str, Any]],
) -> str:
    status = submission_check.get("status")
    feedback = str(submission_check.get("feedback") or "I checked your submission.")
    next_step = str(submission_check.get("next_step") or "Send the next setup step when you are ready.")
    sketch_feedback = _format_sketch_analysis_feedback(sketch_analysis)
    sections: list[str] = []

    if status == "complete":
        sections.append(f"I checked your final answer: {feedback}")
        sections.append(
            f"{next_step}\n\n"
            "I am closing this step-by-step check now. If any part still feels unclear, ask a follow-up question."
        )
    elif status == "correct":
        sections.append(f"I checked your answer: {feedback}")
        sections.append(
            f"Next step: {next_step}\n\n"
            "Try only that next step and send it back. I will keep checking one step at a time without jumping to the full solution."
        )
    elif status == "incorrect":
        sections.append(f"I checked your answer, and this part needs correction: {feedback}")
        sections.append(
            f"Fix this first: {next_step}\n\n"
            "Before moving on, do you understand why this correction is needed? If not, ask why; if yes, send the corrected step."
        )
    elif status == "needs_text":
        sections.append(f"I checked what I can from the sketch: {feedback}")
        sections.append(next_step)
    else:
        sections.append(f"I checked your work: {feedback}")
        sections.append(
            f"Next thing to improve: {next_step}\n\n"
            "Send the corrected setup and I will check it before we move on."
        )

    if sketch_feedback:
        sections.insert(1, sketch_feedback)
    return "\n\n".join(sections)


def _build_hitl_remediation_followup(
    agent_id: str,
    problem: str,
    concept_tag: str,
    student_message: str,
    step_index: int,
    has_drawing: bool = False,
    sketch_analysis: Optional[Dict[str, Any]] = None,
    submission_check: Optional[Dict[str, Any]] = None,
) -> str:
    intent = _remediation_followup_intent(student_message)
    asks_for_next_step = intent["asks_for_next_step"]
    asks_for_explanation = intent["asks_for_explanation"]
    understands = intent["understands"]
    submitted_work = intent["submitted_work"]

    if asks_for_explanation and not asks_for_next_step:
        return (
            "Let's slow down before solving.\n\n"
            f"{_concept_remediation_explanation(agent_id, concept_tag)}\n\n"
            'Does that make sense now? If yes, reply "next step" and I will give only the next setup step.'
        )

    if submitted_work or has_drawing:
        evaluated_submission = submission_check or _check_student_submission(
            agent_id=agent_id,
            concept_tag=concept_tag,
            problem=problem,
            student_message=student_message,
            sketch_analysis=sketch_analysis,
            step_index=step_index,
        )
        return _format_submission_check_response(evaluated_submission, sketch_analysis)

    steps = _concept_next_steps(agent_id, concept_tag)
    completed_step_count = max(0, step_index or 0)
    safe_step_index = max(1, min(completed_step_count + 1, len(steps)))
    next_step = steps[safe_step_index - 1]
    if understands and not asks_for_next_step:
        return (
            "Good. Let's keep the solution hidden for now so you can do the thinking.\n\n"
            f"When you are ready, try this setup step: {next_step}\n\n"
            "Send me your setup or ask for the next step, and I will keep guiding you one step at a time."
        )

    return (
        f"Next step {safe_step_index}: {next_step}\n\n"
        "Try that part and send me your work. I will check it before moving on, without jumping to the full solution."
    )


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Physics Assistant API",
        "version": "2.0.0",
        "docs_url": "/docs",
        "available_agents": ALL_VALID_AGENTS,
        "framework": "strands",
        "courses": {
            "physics_101": PHYSICS_101_AGENTS,
            "physics_102": PHYSICS_102_AGENTS,
            "physics_201": PHYSICS_201_AGENTS,
            "physics_202": PHYSICS_202_AGENTS
        }
    }

@app.post("/agent/create", response_model=AgentCreateResponse)
async def create_agent(request: AgentCreateRequest) -> AgentCreateResponse:
    """
    Create and initialize a physics agent using Strands SDK

    All agents now use Strands SDK with MCP tools and Ollama LLM.
    """
    try:
        logger.info(f"Creating agent: {request.agent_id}")
        
        # Get or create agent using the existing function
        agent = await get_or_create_agent(request.agent_id, request.use_direct_tools, request.enable_rag, request.rag_api_url)
        
        # Get capabilities
        capabilities = await agent.get_capabilities()
        
        return AgentCreateResponse(
            success=True,
            agent_id=request.agent_id,
            message=f"Agent {request.agent_id} created and initialized successfully",
            capabilities=capabilities
        )
        
    except Exception as e:
        logger.error(f"Error creating agent {request.agent_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create agent: {str(e)}"
        )

@app.post("/agent/{agent_id}/solve", response_model=ProblemSolveResponse)
async def solve_problem(
    agent_id: str,
    request: ProblemSolveRequest,
    use_direct_tools: bool = True,
    enable_rag: bool = True,
    rag_api_url: Optional[str] = None
) -> ProblemSolveResponse:
    """
    Solve a physics problem using the specified agent
    """
    request_started = time.perf_counter()
    api_trace_stages: list[Dict[str, Any]] = []
    hitl_trace: Optional[Dict[str, Any]] = None

    try:
        stage_started = time.perf_counter()
        # Validate agent_id
        if agent_id not in ALL_VALID_AGENTS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid agent_id: {agent_id}. Must be one of: {', '.join(ALL_VALID_AGENTS)}"
            )
        _append_perf_stage(api_trace_stages, "validate_agent_id", stage_started)
        
        effective_problem = request.problem
        hitl_question = None
        hitl_result = None
        guidance_prefix = None
        skip_hitl_gate = False
        agent_context = request.context

        kt_response = None
        remediation_followup = None
        class_identifier = None
        if isinstance(request.context, dict):
            kt_response = request.context.get("knowledge_transfer_response")
            remediation_followup = request.context.get("knowledge_transfer_remediation_followup")
            class_identifier = request.context.get("class_identifier")

        if isinstance(remediation_followup, dict):
            stage_started = time.perf_counter()
            concept_tag = str(remediation_followup.get("concept_tag", "")).strip()
            student_message = str(remediation_followup.get("student_message", "")).strip()
            try:
                step_index = int(remediation_followup.get("step_index", 1))
            except (TypeError, ValueError):
                step_index = 1
            original_problem = str(remediation_followup.get("original_problem") or request.problem)
            sketch_analysis = _analyze_student_sketch(remediation_followup.get("drawing"), agent_id, concept_tag)
            has_drawing = bool(remediation_followup.get("has_drawing") or sketch_analysis)
            followup_intent = _remediation_followup_intent(student_message)
            mode = str(remediation_followup.get("mode") or "").strip().lower()
            if mode == "full_solution" or followup_intent["asks_for_full_solution"]:
                effective_problem = original_problem
                skip_hitl_gate = True
                guidance_prefix = (
                    "You asked for the full solution, so I am leaving step-by-step checking and showing the complete worked solution now."
                )
                hitl_result = {
                    "status": "full_solution_requested",
                    "check_id": remediation_followup.get("check_id"),
                    "agent_id": agent_id,
                    "concept_tag": concept_tag,
                    "was_correct": None,
                    "confidence": None,
                    "threshold": None,
                }
                if isinstance(agent_context, dict):
                    agent_context = {
                        key: value
                        for key, value in agent_context.items()
                        if key != "knowledge_transfer_remediation_followup"
                    }
                _append_perf_stage(api_trace_stages, "hitl_full_solution_requested", stage_started)
            else:
                submission_check = None
                if followup_intent["submitted_work"] or has_drawing:
                    submission_check = _check_student_submission(
                        agent_id=agent_id,
                        concept_tag=concept_tag,
                        problem=original_problem,
                        student_message=student_message,
                        sketch_analysis=sketch_analysis,
                        step_index=step_index,
                    )
                response_step_index = max(0, step_index)
                should_advance_step = (
                    submission_check
                    and submission_check.get("status") in {"correct", "complete"}
                    and submission_check.get("advance_step", True) is not False
                )
                if should_advance_step:
                    response_step_index = max(0, step_index + 1)
                response_hitl_status = (
                    "remediation_complete"
                    if submission_check and submission_check.get("status") == "complete"
                    else "remediation_followup"
                )
                guidance = _build_hitl_remediation_followup(
                    agent_id=agent_id,
                    problem=original_problem,
                    concept_tag=concept_tag,
                    student_message=student_message,
                    step_index=step_index,
                    has_drawing=has_drawing,
                    sketch_analysis=sketch_analysis,
                    submission_check=submission_check,
                )
                hitl_payload = {
                    "status": response_hitl_status,
                    "check_id": remediation_followup.get("check_id"),
                    "agent_id": agent_id,
                    "concept_tag": concept_tag,
                    "step_index": response_step_index,
                    "student_message": student_message,
                    "has_drawing": has_drawing,
                }
                if sketch_analysis:
                    hitl_payload["sketch_analysis"] = sketch_analysis
                if submission_check:
                    hitl_payload["submission_check"] = submission_check
                _append_perf_stage(api_trace_stages, "early_return_hitl_remediation_followup", stage_started)
                api_trace = _finalize_perf_trace("api_route", api_trace_stages, request_started)
                performance_trace = _merge_performance_trace(api_trace=api_trace)
                _log_slow_trace(agent_id, request.user_id, performance_trace)
                return ProblemSolveResponse(
                    success=True,
                    agent_id=agent_id,
                    problem=original_problem,
                    solution=guidance,
                    hitl=hitl_payload,
                    metadata={
                        "hitl": hitl_payload,
                        "framework": "strands",
                        "performance_trace": performance_trace,
                    },
                )

        if not skip_hitl_gate and knowledge_transfer_gate and knowledge_transfer_gate.is_enabled_for(agent_id):
            if isinstance(kt_response, dict):
                check_id = str(kt_response.get("check_id", "")).strip()
                selected_option_id = str(kt_response.get("selected_option_id", "")).strip()
                if check_id and selected_option_id:
                    stage_started = time.perf_counter()
                    hitl_result, hitl_trace = knowledge_transfer_gate.process_answer_with_trace(
                        check_id=check_id,
                        selected_option_id=selected_option_id,
                        user_id=request.user_id or "react_user",
                        session_id=request.session_id,
                        class_identifier=class_identifier,
                    )
                    _append_perf_stage(
                        api_trace_stages,
                        "hitl_process_answer",
                        stage_started,
                        answer_processed=bool(hitl_result),
                    )
                    if hitl_result:
                        effective_problem = str(hitl_result.get("original_problem", request.problem))
                        guidance_prefix = str(hitl_result.get("guidance", "")).strip()
                        if hitl_result.get("status") == "remediation_required" or hitl_result.get("was_correct") is False:
                            stage_started = time.perf_counter()
                            hitl_payload = {
                                "status": hitl_result.get("status", "remediation_required"),
                                "check_id": hitl_result.get("check_id"),
                                "agent_id": hitl_result.get("agent_id", agent_id),
                                "concept_tag": hitl_result.get("concept_tag"),
                                "was_correct": hitl_result.get("was_correct"),
                                "confidence": hitl_result.get("confidence"),
                                "threshold": hitl_result.get("threshold"),
                                "guidance": guidance_prefix,
                                "remediation": hitl_result.get("remediation"),
                            }
                            _append_perf_stage(api_trace_stages, "early_return_hitl_remediation", stage_started)
                            api_trace = _finalize_perf_trace("api_route", api_trace_stages, request_started)
                            performance_trace = _merge_performance_trace(api_trace=api_trace, hitl_trace=hitl_trace)
                            _log_slow_trace(agent_id, request.user_id, performance_trace)
                            return ProblemSolveResponse(
                                success=True,
                                agent_id=agent_id,
                                problem=effective_problem,
                                solution=guidance_prefix,
                                hitl=hitl_payload,
                                metadata={
                                    "hitl": hitl_payload,
                                    "framework": "strands",
                                    "performance_trace": performance_trace,
                                },
                            )
                    else:
                        logger.error(
                            "HITL_GATE_FALLBACK: invalid/expired check_id provided; proceeding without gating. "
                            "agent=%s user=%s check_id=%s",
                            agent_id,
                            request.user_id,
                            check_id,
                        )
                else:
                    logger.error(
                        "HITL_GATE_FALLBACK: malformed knowledge_transfer_response payload; proceeding without gating. "
                        "agent=%s user=%s",
                        agent_id,
                        request.user_id,
                    )
            else:
                stage_started = time.perf_counter()
                hitl_question, hitl_trace = await knowledge_transfer_gate.maybe_create_check_with_trace(
                    agent_id=agent_id,
                    problem=request.problem,
                    user_id=request.user_id or "react_user",
                    session_id=request.session_id,
                    class_identifier=class_identifier,
                )
                _append_perf_stage(
                    api_trace_stages,
                    "hitl_maybe_create_check",
                    stage_started,
                    question_required=bool(hitl_question),
                )
                if hitl_question:
                    stage_started = time.perf_counter()
                    _append_perf_stage(api_trace_stages, "early_return_hitl_question", stage_started)
                    api_trace = _finalize_perf_trace("api_route", api_trace_stages, request_started)
                    performance_trace = _merge_performance_trace(api_trace=api_trace, hitl_trace=hitl_trace)
                    _log_slow_trace(agent_id, request.user_id, performance_trace)
                    return ProblemSolveResponse(
                        success=True,
                        agent_id=agent_id,
                        problem=request.problem,
                        solution="",
                        hitl=hitl_question,
                        metadata={
                            "hitl": hitl_question,
                            "framework": "strands",
                            "performance_trace": performance_trace,
                        },
                    )

        # Get or create agent
        stage_started = time.perf_counter()
        agent = await get_or_create_agent(agent_id, use_direct_tools, enable_rag, rag_api_url)
        _append_perf_stage(api_trace_stages, "get_or_create_agent", stage_started)

        logger.info(f"Solving problem with {agent_id}: {effective_problem[:50]}...")

        # Solve the problem with user and session context for database logging
        stage_started = time.perf_counter()
        result = await agent.solve_problem(
            problem=effective_problem,
            context=agent_context,
            user_id=request.user_id,
            session_id=request.session_id
        )
        _append_perf_stage(
            api_trace_stages,
            "agent_solve_problem",
            stage_started,
            success=bool(result.get("success")),
        )

        if result.get("success") and guidance_prefix and result.get("solution"):
            result["solution"] = f"{guidance_prefix}\n\n{result['solution']}"

        stage_started = time.perf_counter()
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        agent_trace = metadata.get("performance_trace") if isinstance(metadata.get("performance_trace"), dict) else None
        if hitl_result:
            metadata["hitl"] = {
                "status": hitl_result.get("status", "answer_processed"),
                "check_id": hitl_result.get("check_id"),
                "agent_id": hitl_result.get("agent_id", agent_id),
                "concept_tag": hitl_result.get("concept_tag"),
                "was_correct": hitl_result.get("was_correct"),
                "confidence": hitl_result.get("confidence"),
                "threshold": hitl_result.get("threshold"),
            }
            result["hitl"] = metadata["hitl"]
        elif hitl_question:
            metadata["hitl"] = hitl_question
            result["hitl"] = hitl_question
        _append_perf_stage(api_trace_stages, "response_metadata_assembly", stage_started)

        api_trace = _finalize_perf_trace("api_route", api_trace_stages, request_started)
        metadata["performance_trace"] = _merge_performance_trace(
            api_trace=api_trace,
            hitl_trace=hitl_trace,
            agent_trace=agent_trace,
        )
        _log_slow_trace(agent_id, request.user_id, metadata["performance_trace"])
        if metadata:
            result["metadata"] = metadata

        return ProblemSolveResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error solving problem with {agent_id}: {str(e)}")
        api_trace = _finalize_perf_trace("api_route", api_trace_stages, request_started)
        performance_trace = _merge_performance_trace(api_trace=api_trace, hitl_trace=hitl_trace)
        _log_slow_trace(agent_id, request.user_id, performance_trace)
        return ProblemSolveResponse(
            success=False,
            agent_id=agent_id,
            problem=request.problem,
            error=str(e),
            metadata={"performance_trace": performance_trace},
        )

@app.get("/agent/{agent_id}/health", response_model=AgentHealthResponse)
async def check_agent_health(
    agent_id: str,
    use_direct_tools: bool = True
) -> AgentHealthResponse:
    """
    Check the health status of a physics agent
    """
    try:
        # Validate agent_id
        if agent_id not in ALL_VALID_AGENTS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid agent_id: {agent_id}. Must be one of: {', '.join(ALL_VALID_AGENTS)}"
            )
        
        # Get or create agent (use defaults for RAG)
        agent = await get_or_create_agent(agent_id, use_direct_tools)

        # Get health status
        health = await agent.health_check()
        
        return AgentHealthResponse(**health)
        
    except Exception as e:
        logger.error(f"Error checking health for {agent_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check agent health: {str(e)}"
        )

@app.get("/agent/{agent_id}/capabilities")
async def get_agent_capabilities(
    agent_id: str,
    use_direct_tools: bool = True
) -> Dict[str, Any]:
    """
    Get the capabilities of a physics agent
    """
    try:
        # Validate agent_id
        if agent_id not in ALL_VALID_AGENTS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid agent_id: {agent_id}. Must be one of: {', '.join(ALL_VALID_AGENTS)}"
            )
        
        # Get or create agent (use defaults for RAG)
        agent = await get_or_create_agent(agent_id, use_direct_tools)

        # Get capabilities
        capabilities = await agent.get_capabilities()
        
        return capabilities
        
    except Exception as e:
        logger.error(f"Error getting capabilities for {agent_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get agent capabilities: {str(e)}"
        )

@app.get("/agents/list")
async def list_available_agents():
    """
    List all available physics agents organized by course level
    """
    return {
        "available_agents": [
            # Physics 101 - Mechanics (LangChain-based)
            {
                "agent_id": "forces_agent",
                "name": "Forces Agent",
                "description": "Handles force analysis, free body diagrams, and Newton's laws",
                "course": "Physics 101",
                "framework": "langchain"
            },
            {
                "agent_id": "kinematics_agent",
                "name": "Kinematics Agent",
                "description": "Handles motion analysis, projectile motion, and kinematics equations",
                "course": "Physics 101",
                "framework": "langchain"
            },
            {
                "agent_id": "math_agent",
                "name": "Math Agent",
                "description": "Handles physics algebra practice, mathematical calculations, and computational problems",
                "course": "All",
                "framework": "langchain"
            },
            {
                "agent_id": "momentum_agent",
                "name": "Momentum Agent",
                "description": "Handles momentum, impulse, and collision problems",
                "course": "Physics 101",
                "framework": "langchain"
            },
            {
                "agent_id": "energy_agent",
                "name": "Energy Agent",
                "description": "Handles work, energy, power, and conservation of energy problems",
                "course": "Physics 101",
                "framework": "langchain"
            },
            {
                "agent_id": "angular_motion_agent",
                "name": "Angular Motion Agent",
                "description": "Handles rotational motion, angular momentum, and torque problems",
                "course": "Physics 101",
                "framework": "langchain"
            },
            # Physics 102 - Thermodynamics & Waves (Strands-based)
            {
                "agent_id": "thermodynamics_agent",
                "name": "Thermodynamics Agent",
                "description": "Handles ideal gas law, heat transfer, thermal expansion, and Carnot efficiency",
                "course": "Physics 102",
                "framework": "strands"
            },
            {
                "agent_id": "waves_agent",
                "name": "Waves Agent",
                "description": "Handles wave mechanics, Doppler effect, sound intensity, and interference",
                "course": "Physics 102",
                "framework": "strands"
            },
            # Physics 201 - Electricity & Magnetism (Strands-based)
            {
                "agent_id": "electromagnetism_agent",
                "name": "Electromagnetism Agent",
                "description": "Handles Coulomb's law, circuits, magnetic fields, and Faraday's law",
                "course": "Physics 201",
                "framework": "strands"
            },
            # Physics 202 - Optics & Modern Physics (Strands-based)
            {
                "agent_id": "optics_agent",
                "name": "Optics Agent",
                "description": "Handles refraction, lenses, mirrors, diffraction, and interference",
                "course": "Physics 202",
                "framework": "strands"
            },
            {
                "agent_id": "modern_physics_agent",
                "name": "Modern Physics Agent",
                "description": "Handles relativity, quantum mechanics, and nuclear physics",
                "course": "Physics 202",
                "framework": "strands"
            }
        ],
        "active_agents": list(agent_store.keys()),
        "agent_counts": {
            "physics_101": len(PHYSICS_101_AGENTS),
            "physics_102": len(PHYSICS_102_AGENTS),
            "physics_201": len(PHYSICS_201_AGENTS),
            "physics_202": len(PHYSICS_202_AGENTS),
            "total": len(ALL_VALID_AGENTS)
        }
    }
# @app.get("/agents/list")
# async def list_available_agents():
#     return {
#         "available_agents": list(Config.PHYSICS_AGENTS.values()),
#         "active_agents": list(agent_store.keys())
#     }

@app.delete("/agent/{agent_id}")
async def remove_agent(agent_id: str, use_direct_tools: bool = True):
    """
    Remove an agent from the store
    """
    agent_key = f"{agent_id}_{use_direct_tools}"
    
    if agent_key in agent_store:
        del agent_store[agent_key]
        return {"message": f"Agent {agent_id} removed successfully"}
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Agent {agent_id} not found"
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    API health check endpoint
    """
    return {
        "status": "healthy",
        "active_agents": len(agent_store),
        "agent_keys": list(agent_store.keys())
    }

# RAG Configuration Endpoints
@app.get("/rag/status")
async def get_rag_status():
    """
    Get RAG system status and configuration
    """
    try:
        # Check how many agents have RAG enabled
        rag_enabled_agents = [
            key for key in agent_store.keys() 
            if key.split("_")[-1] == "True"  # enable_rag is the last part of the key
        ]
        
        return {
            "status": "available",
            "rag_enabled_agents": len(rag_enabled_agents),
            "total_agents": len(agent_store),
            "rag_endpoints": [
                "/rag/query",
                "/rag/semantic-search", 
                "/rag/graph-search",
                "/rag/learning-path"
            ]
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/rag/clear-cache")
async def clear_rag_cache():
    """
    Clear RAG client caches for all agents
    """
    try:
        cleared_count = 0
        for agent in agent_store.values():
            if hasattr(agent, 'rag_client') and agent.rag_client:
                agent.rag_client.clear_cache()
                cleared_count += 1
        
        return {
            "status": "success",
            "message": f"RAG cache cleared for {cleared_count} agents"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear RAG cache: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
