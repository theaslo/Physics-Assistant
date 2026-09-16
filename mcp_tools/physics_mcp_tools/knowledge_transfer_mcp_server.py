#!/usr/bin/env python3
"""
Knowledge Transfer MCP Server
Provides curated multiple-choice knowledge checks for HITL pedagogy gating.
"""

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional

import aiohttp
from fastmcp import FastMCP


NAME = "Knowledge Transfer MCP Server"
logger = logging.getLogger(__name__)


def _normalize_agent_type(agent_id: str) -> str:
    value = (agent_id or "").strip().lower()
    if value.endswith("_agent"):
        value = value[:-6]
    if value == "angular_motion":
        return "angular_motion"
    if value in {"forces", "kinematics", "energy", "momentum", "math"}:
        return value
    return value or "forces"


def _fallback_questions() -> Dict[str, Dict[str, Dict[str, Any]]]:
    return {
        "forces": {
            "newton_first_law": {
                "question_text": "Which statement best describes Newton's First Law?",
                "options": [
                    {"id": "A", "text": "An object changes velocity only when a net external force acts on it.", "is_correct": True, "feedback": "Correct: inertia means no change in motion without net force."},
                    {"id": "B", "text": "A moving object always needs a force to keep moving.", "is_correct": False, "feedback": "Not correct: constant velocity requires zero net force."},
                    {"id": "C", "text": "Heavier objects always resist motion changes less.", "is_correct": False, "feedback": "Not correct: larger mass means more inertia, not less."},
                ],
                "explanation_correct": "Newton's First Law is the inertia law: no net force implies constant velocity.",
                "explanation_incorrect": "Key idea: motion changes only when net external force is nonzero.",
            },
            "newton_second_law": {
                "question_text": "For Newton's Second Law, what equation links net force, mass, and acceleration?",
                "options": [
                    {"id": "A", "text": "F = m a", "is_correct": True, "feedback": "Correct: net force equals mass times acceleration."},
                    {"id": "B", "text": "F = m / a", "is_correct": False, "feedback": "Not correct: acceleration multiplies mass, it does not divide it."},
                    {"id": "C", "text": "F = a / m", "is_correct": False, "feedback": "Not correct: this reverses the physical relationship."},
                ],
                "explanation_correct": "Use the vector form of Newton's Second Law: ΣF = m a.",
                "explanation_incorrect": "Use ΣF = m a. Acceleration is proportional to net force and inversely related to mass.",
            },
            "newton_third_law": {
                "question_text": "Which pair is a Newton's Third Law action-reaction pair?",
                "options": [
                    {"id": "A", "text": "Earth pulls on ball and ball pulls on Earth with equal magnitude opposite direction.", "is_correct": True, "feedback": "Correct: equal and opposite forces on different bodies."},
                    {"id": "B", "text": "Weight and normal force on the same block.", "is_correct": False, "feedback": "Not correct: those are not a third-law pair; they act on the same object."},
                    {"id": "C", "text": "Friction and acceleration of one object.", "is_correct": False, "feedback": "Not correct: acceleration is not a force."},
                ],
                "explanation_correct": "Third-law forces are equal-opposite and act on two different interacting bodies.",
                "explanation_incorrect": "Third-law pairs act on different objects, not the same object.",
            },
            "hookes_law": {
                "question_text": "For an ideal spring near equilibrium, which relation is Hooke's law (magnitude form)?",
                "options": [
                    {"id": "A", "text": "F = k x", "is_correct": True, "feedback": "Correct for magnitudes; vector form is F = -k x."},
                    {"id": "B", "text": "F = x / k", "is_correct": False, "feedback": "Not correct: this inverts the spring constant relationship."},
                    {"id": "C", "text": "F = k / x", "is_correct": False, "feedback": "Not correct and nonphysical near x=0."},
                ],
                "explanation_correct": "Magnitude relation is |F| = k|x|, and restoring direction gives vector form F = -k x.",
                "explanation_incorrect": "Hooke's law is linear in displacement: force magnitude grows with |x|.",
            },
            "incline_components": {
                "question_text": "On an incline at angle θ, which is the component of weight parallel to plane (down slope)?",
                "options": [
                    {"id": "A", "text": "W sin(θ)", "is_correct": True, "feedback": "Correct: W_parallel = mg sin θ."},
                    {"id": "B", "text": "W cos(θ)", "is_correct": False, "feedback": "Not correct: W cos θ is the perpendicular component."},
                    {"id": "C", "text": "W tan(θ)", "is_correct": False, "feedback": "Not correct: tan θ is not used for direct component magnitude."},
                ],
                "explanation_correct": "Resolve weight into slope-aligned axes: parallel is mg sin θ, perpendicular is mg cos θ.",
                "explanation_incorrect": "Use axes aligned with the incline: parallel weight is mg sin θ.",
            },
        },
        "kinematics": {
            "projectile_components": {
                "question_text": "For projectile launch speed v0 at angle θ, what is the initial vertical component?",
                "options": [
                    {"id": "A", "text": "v0 sin(θ)", "is_correct": True, "feedback": "Correct: vertical launch component uses sine."},
                    {"id": "B", "text": "v0 cos(θ)", "is_correct": False, "feedback": "Not correct for vertical; cosine is horizontal component."},
                    {"id": "C", "text": "v0 tan(θ)", "is_correct": False, "feedback": "Not correct: tan θ is a ratio, not a component projection."},
                ],
                "explanation_correct": "Use component decomposition: v0x = v0 cos θ, v0y = v0 sin θ.",
                "explanation_incorrect": "Vertical and horizontal components use sine/cosine respectively.",
            },
            "projectile_peak": {
                "question_text": "At maximum height of a projectile (neglecting air resistance), which is true?",
                "options": [
                    {"id": "A", "text": "Vertical velocity vy = 0 while horizontal velocity remains nonzero.", "is_correct": True, "feedback": "Correct: vy crosses zero at peak, vx remains constant."},
                    {"id": "B", "text": "Both vx and vy are zero.", "is_correct": False, "feedback": "Not correct: horizontal speed persists without horizontal acceleration."},
                    {"id": "C", "text": "Acceleration is zero at peak.", "is_correct": False, "feedback": "Not correct: acceleration remains downward at g."},
                ],
                "explanation_correct": "Only vy becomes zero at the top; acceleration is still -g downward.",
                "explanation_incorrect": "At peak, vy=0 but ax=0 and ay=-g still hold.",
            },
            "equation_selection": {
                "question_text": "Which constant-acceleration equation directly avoids time t?",
                "options": [
                    {"id": "A", "text": "v_f^2 = v_0^2 + 2 a Δx", "is_correct": True, "feedback": "Correct: this relation eliminates time."},
                    {"id": "B", "text": "v_f = v_0 + a t", "is_correct": False, "feedback": "Not correct: this equation includes time explicitly."},
                    {"id": "C", "text": "x = x_0 + v_0 t + (1/2) a t^2", "is_correct": False, "feedback": "Not correct: this also contains time."},
                ],
                "explanation_correct": "Use v_f^2 = v_0^2 + 2aΔx when time is unknown or intentionally eliminated.",
                "explanation_incorrect": "Only the squared-velocity relation eliminates time directly.",
            },
            "sign_convention": {
                "question_text": "If +y is upward in projectile motion near Earth, what is acceleration ay?",
                "options": [
                    {"id": "A", "text": "-g", "is_correct": True, "feedback": "Correct: gravity points downward."},
                    {"id": "B", "text": "+g", "is_correct": False, "feedback": "Not correct under +y upward convention."},
                    {"id": "C", "text": "0", "is_correct": False, "feedback": "Not correct: gravity is present throughout flight."},
                ],
                "explanation_correct": "Choose signs from axis convention: with +y upward, ay = -g.",
                "explanation_incorrect": "Keep sign convention consistent; ay is negative when +y is upward.",
            },
            "range_formula": {
                "question_text": "For launch and landing at same height, projectile range is:",
                "options": [
                    {"id": "A", "text": "R = v0^2 sin(2θ) / g", "is_correct": True, "feedback": "Correct for same launch/landing height."},
                    {"id": "B", "text": "R = v0 sin(θ) / g", "is_correct": False, "feedback": "Not correct dimensionally for range distance."},
                    {"id": "C", "text": "R = 2 v0 cos(θ) / g", "is_correct": False, "feedback": "Not correct: missing required speed factor and angle coupling."},
                ],
                "explanation_correct": "For level-to-level projectile motion, R = (v0^2 sin 2θ)/g.",
                "explanation_incorrect": "Use full level-ground range formula with sin(2θ).",
            },
        },
    }


async def _fetch_question_from_database(
    agent_type: str,
    concept_tag: Optional[str],
) -> Optional[Dict[str, Any]]:
    host = os.getenv("DATABASE_API_HOST", "database-api")
    port = int(os.getenv("DATABASE_API_PORT", "8001"))
    base_url = f"http://{host}:{port}"
    url = f"{base_url}/knowledge-transfer/questions/pick"
    params = {"agent_type": agent_type}
    if concept_tag:
        params["concept_tag"] = concept_tag

    timeout = aiohttp.ClientTimeout(total=6)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return None
                payload = await resp.json()
                if not isinstance(payload, dict):
                    return None
                if not payload.get("question"):
                    return None
                return payload
    except Exception:
        return None


def _pick_fallback_question(agent_type: str, concept_tag: Optional[str]) -> Dict[str, Any]:
    questions = _fallback_questions()
    agent_bank = questions.get(agent_type) or questions["forces"]
    concept = concept_tag if concept_tag in agent_bank else next(iter(agent_bank.keys()))
    selected = dict(agent_bank[concept])
    selected["concept_tag"] = concept
    selected["agent_type"] = agent_type
    selected["question_id"] = None
    return selected


def serve(host: str = "localhost", port: int = 10112, transport: str = "streamable_http"):
    mcp = FastMCP(NAME)

    @mcp.tool()
    async def get_guided_knowledge_question(request_json: str) -> str:
        """
        Fetch a curated knowledge-check multiple-choice question.

        Args:
            request_json: JSON string with
                {
                  "agent_id": "forces_agent|kinematics_agent",
                  "concept_tag": "optional concept tag",
                  "user_prompt": "original prompt"
                }

        Returns:
            String containing KT_JSON_START/KT_JSON_END wrapped JSON payload.
        """
        try:
            data = json.loads(request_json) if isinstance(request_json, str) else request_json
            agent_id = str(data.get("agent_id", "forces_agent"))
            concept_tag = data.get("concept_tag")

            agent_type = _normalize_agent_type(agent_id)
            question_row = await _fetch_question_from_database(agent_type, concept_tag)
            if question_row is None:
                question_row = _pick_fallback_question(agent_type, concept_tag)

            options = question_row.get("options", [])
            payload = {
                "type": "knowledge_check",
                "agent_id": agent_id,
                "agent_type": agent_type,
                "concept_tag": question_row.get("concept_tag", concept_tag),
                "question_id": question_row.get("question_id"),
                "question": question_row.get("question") or question_row.get("question_text"),
                "options": [{"id": str(opt["id"]), "text": str(opt["text"])} for opt in options],
                "correct_option_id": question_row.get("correct_option_id"),
                "correct_feedback": question_row.get("correct_feedback") or question_row.get("explanation_correct"),
                "incorrect_feedback": question_row.get("incorrect_feedback") or question_row.get("explanation_incorrect"),
                "distractor_feedback": question_row.get("distractor_feedback", {}),
            }

            return f"KT_JSON_START\n{json.dumps(payload)}\nKT_JSON_END"
        except Exception as e:
            logger.exception("Knowledge question retrieval failed")
            return f"Error retrieving knowledge question: {str(e)}"

    logger.info(f"{NAME} at {host}:{port} using {transport}")
    if transport == "sse":
        mcp.sse_http_app.run(host=host, port=port)
    if transport == "streamable_http":
        import uvicorn
        uvicorn.run(mcp.streamable_http_app, host=host, port=port)


def main():
    parser = argparse.ArgumentParser(description="Run Knowledge Transfer MCP Server")
    parser.add_argument("--run", default="mcp-server", help="Command to run")
    parser.add_argument("--host", default="localhost", help="Host to bind server to")
    parser.add_argument("--port", type=int, default=10112, help="Port to bind server to")
    parser.add_argument("--transport", default="streamable_http", help="Transport type")
    args = parser.parse_args()

    if args.run == "mcp-server":
        serve(args.host, args.port, args.transport)
    else:
        raise ValueError(f"Unknown run option: {args.run}")


if __name__ == "__main__":
    main()
